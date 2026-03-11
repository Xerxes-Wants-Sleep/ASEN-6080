import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

sys.path.append("../../")

from prob1b import (
    convert_filter_output_km_to_m_for_plotting,
    load_problem2_inputs,
    make_trace_cov_pos_plot,
    make_trace_cov_vel_plot,
    rms_by_component_2d,
)
from src.Functions.range_rangerate import H_range_rangerate
from src.Functions.srif import (
    SquareRootInformationFilter,
    _info_factor_from_cov,
    _qr_householder_transform,
)
from src.helpers.plotting.common import as_hours, savefig
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot


class SquareRootInformationFilterEq51023(SquareRootInformationFilter):
    """
    SRIF variant for HW5-1c:
      - Time update uses Eq. 5.10.23 directly:
            R_minus = R_prev * Phi^{-1}
      - Does NOT re-triangularize R_minus before measurement update.
    """

    def run(self, all_meas, stations, Xtrue_meas: np.ndarray | None = None):
        if self.process_noise_mode != "none":
            raise ValueError("HW5-1c script assumes process_noise_mode='none'.")

        station_map = {st.name: st for st in stations}
        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
        t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
        st_meas = [m["station"] for m in all_meas]
        N = len(all_meas)

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (N, 6):
                raise ValueError(f"Xtrue_meas must have shape ({N}, 6) aligned with sorted measurement order.")

        Xstar_hist, Phi_i0_hist = self.propagate_state_and_stm_history(t_meas)
        Phi_step = self.phi_i0_to_phi_step(Phi_i0_hist)

        V = la.cholesky(self.R, lower=True, check_finite=False)

        residuals = np.full((N, 2), np.nan, dtype=float)
        postfit_lin = np.full((N, 2), np.nan, dtype=float)
        postfit_nl = np.full((N, 2), np.nan, dtype=float)  # placeholder for plotting compatibility

        X_pf = np.full((N, 6), np.nan, dtype=float)
        P_meas = np.full((N, 6, 6), np.nan, dtype=float)
        P_pf = np.full((N, 36), np.nan, dtype=float)
        two_sigma = np.full((N, 6), np.nan, dtype=float)
        state_error = None if Xtrue_meas is None else np.full((N, 6), np.nan, dtype=float)

        prefit_whitened = np.full((N, 2), np.nan, dtype=float)
        postfit_whitened = np.full((N, 2), np.nan, dtype=float)
        P_pred_hist = np.full((N, 6, 6), np.nan, dtype=float)
        x_pred_hist = np.full((N, 6), np.nan, dtype=float)

        rminus_lower_norm = np.full(N, np.nan, dtype=float)
        rminus_rel_lower_norm = np.full(N, np.nan, dtype=float)

        x_hat = np.zeros(6, dtype=float)
        R_info = _info_factor_from_cov(self.P0)
        b_info = R_info @ x_hat

        for k in range(N):
            t = float(t_meas[k])
            st = station_map[st_meas[k]]
            Xstar = Xstar_hist[k, :]
            Phi = Phi_step[k, :, :]

            # Eq. 5.10.23: R_minus = R_prev * Phi^{-1}, no re-triangularization here.
            Rminus = la.solve(Phi.T, R_info.T, check_finite=False).T
            bminus = b_info.copy()

            # quantify how non-triangular R_minus becomes
            lower = np.tril(Rminus, k=-1)
            rminus_lower_norm[k] = float(np.linalg.norm(lower))
            rminus_rel_lower_norm[k] = float(np.linalg.norm(lower) / max(np.linalg.norm(Rminus), 1e-30))

            # Rminus is not triangular; recover covariance with a full solve.
            I6 = np.eye(6)
            Rminus_inv = la.solve(Rminus, I6, check_finite=False)
            P_pred = Rminus_inv @ Rminus_inv.T
            P_pred_hist[k, :, :] = P_pred

            # R_minus is not triangular in general, so use full linear solve.
            x_pred = la.solve(Rminus, bminus, check_finite=False)
            x_pred_hist[k, :] = x_pred

            Y = np.array([all_meas[k]["rho_km"], all_meas[k]["rho_dot_km_s"]], dtype=float)
            C_ref = self.G(st, Xstar, t)

            if C_ref is None:
                x_hat = x_pred
                R_info = Rminus
                b_info = bminus
                X_post = Xstar + x_hat

                X_pf[k, :] = X_post
                I6 = np.eye(6)
                Rinfo_inv = la.solve(R_info, I6, check_finite=False)
                P_post = Rinfo_inv @ Rinfo_inv.T
                P_meas[k, :, :] = P_post
                P_pf[k, :] = P_post.reshape(-1, order="F")
                two_sigma[k, :] = 2.0 * np.sqrt(np.maximum(np.diag(P_post), 0.0))

                if state_error is not None:
                    state_error[k, :] = X_post - Xtrue_meas[k, :]

                self.Xhat = X_post
                self.Phat = P_post
                self.log_epoch(
                    t,
                    postfit_resid=np.array([np.nan, np.nan], dtype=float),
                    Xtrue=(None if Xtrue_meas is None else Xtrue_meas[k, :]),
                )
                continue

            OminusC = Y - C_ref
            residuals[k, :] = OminusC

            y = la.solve_triangular(V, OminusC, lower=True, check_finite=False)
            prefit_whitened[k, :] = y

            Rs, Vs, _ = st.ecef2eci(t, st.r_ecef, np.zeros(3))
            H = H_range_rangerate(Xstar[:3], Xstar[3:], Rs, Vs)
            Htilde = la.solve_triangular(V, H, lower=True, check_finite=False)

            M = np.vstack(
                [
                    np.hstack([Rminus, bminus.reshape(-1, 1)]),
                    np.hstack([Htilde, y.reshape(-1, 1)]),
                ]
            )
            out = _qr_householder_transform(M, fix_sign=True)

            R_post = out[:6, :6]
            b_post = out[:6, 6]
            e = out[6:, 6]
            postfit_whitened[k, :] = e

            x_hat = la.solve_triangular(R_post, b_post, lower=False, check_finite=False)

            R_info = R_post
            b_info = b_post

            X_post = Xstar + x_hat
            X_pf[k, :] = X_post

            # Here R_info is upper-triangular after measurement QR, but full solve is fine.
            I6 = np.eye(6)
            Rinfo_inv = la.solve(R_info, I6, check_finite=False)
            P_post = Rinfo_inv @ Rinfo_inv.T
            P_meas[k, :, :] = P_post
            P_pf[k, :] = P_post.reshape(-1, order="F")
            two_sigma[k, :] = 2.0 * np.sqrt(np.maximum(np.diag(P_post), 0.0))
            postfit_lin[k, :] = V @ e

            if state_error is not None:
                state_error[k, :] = X_post - Xtrue_meas[k, :]

            self.Xhat = X_post
            self.Phat = P_post
            self.log_epoch(t, postfit_resid=postfit_lin[k, :], Xtrue=(None if Xtrue_meas is None else Xtrue_meas[k, :]))

        rms_final = self.print_rms_summary(label="SRIF Eq5.10.23", first_pass_gap_s=self.first_pass_gap_s)

        return {
            "t_meas": t_meas,
            "station_meas": st_meas,
            "xhat_meas": X_pf,
            "Xhat_meas": X_pf,
            "X_pf": X_pf,
            "prefit_resids_final": residuals,
            "postfit_resids_linear_final": postfit_lin,
            "postfit_resids_meas": postfit_nl,
            "prefit_res_whitened": prefit_whitened,
            "postfit_res_whitened": postfit_whitened,
            "P_meas": P_meas,
            "Phat_meas": P_meas,
            "P_pf": P_pf,
            "P_pred_hist": P_pred_hist,
            "x_pred_hist": x_pred_hist,
            "Phi_step": Phi_step,
            "two_sigma_meas": two_sigma,
            "state_error_meas": state_error,
            "Rminus_lower_norm": rminus_lower_norm,
            "Rminus_rel_lower_norm": rminus_rel_lower_norm,
            "rms_final": rms_final,
            "rms_by_iter": None,
        }


def run_srif_standard(*, all_meas, stations, Xtrue_meas, x0_bar, P0, R, mu, J2, J3):
    srif = SquareRootInformationFilter(
        X0_star=x0_bar,
        P0=P0,
        R=R,
        Q=np.zeros((6, 6), dtype=float),
        mu=mu,
        J2=J2,
        J3=J3,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
        j2=True,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
        process_noise_mode="none",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return srif.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)


def run_srif_eq51023(*, all_meas, stations, Xtrue_meas, x0_bar, P0, R, mu, J2, J3):
    srif = SquareRootInformationFilterEq51023(
        X0_star=x0_bar,
        P0=P0,
        R=R,
        Q=np.zeros((6, 6), dtype=float),
        mu=mu,
        J2=J2,
        J3=J3,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
        j2=True,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
        process_noise_mode="none",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return srif.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)


def make_state_error_difference_plot(t_s: np.ndarray, de_m: np.ndarray, outdir: Path):
    t_hr = as_hours(t_s)
    labels = ["dx [m]", "dy [m]", "dz [m]", "dvx [m/s]", "dvy [m/s]", "dvz [m/s]"]

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axs = axs.flatten()
    for i in range(6):
        axs[i].plot(t_hr, de_m[:, i], ".", markersize=2)
        axs[i].axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
    axs[-2].set_xlabel("Time [hours]")
    axs[-1].set_xlabel("Time [hours]")
    fig.suptitle("State-Error Difference: Eq5.10.23 SRIF - Standard SRIF")
    savefig(fig, outdir / "state_error_difference_eq51023_minus_standard.png")


def make_rminus_nontriangular_plot(t_s: np.ndarray, lower_norm: np.ndarray, rel_lower_norm: np.ndarray, outdir: Path):
    t_hr = as_hours(t_s)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].semilogy(t_hr, np.maximum(lower_norm, 1e-30), ".", markersize=2)
    ax[0].set_ylabel("||tril(R-, -1)||")
    ax[0].set_title("Non-triangularity of R- (Eq. 5.10.23 run)")
    ax[0].grid(True, alpha=0.3)

    ax[1].semilogy(t_hr, np.maximum(rel_lower_norm, 1e-30), ".", markersize=2)
    ax[1].set_ylabel("||lower|| / ||R-||")
    ax[1].set_xlabel("Time [hours]")
    ax[1].grid(True, alpha=0.3)

    savefig(fig, outdir / "rminus_nontriangularity.png")


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0_bar = load_problem2_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    out_std = run_srif_standard(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0_bar=x0_bar,
        P0=P0,
        R=R,
        mu=mu,
        J2=J2,
        J3=J3,
    )
    out_eq = run_srif_eq51023(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0_bar=x0_bar,
        P0=P0,
        R=R,
        mu=mu,
        J2=J2,
        J3=J3,
    )

    res_std = convert_filter_output_km_to_m_for_plotting(out_std, R_km=R)
    res_eq = convert_filter_output_km_to_m_for_plotting(out_eq, R_km=R)

    plot_root = Path(__file__).resolve().parent / "Plots" / "Prob1c"
    std_dir = plot_root / "SRIF_Standard"
    eq_dir = plot_root / "SRIF_Eq51023_NoTriRminus"
    cmp_dir = plot_root / "Comparison"
    std_dir.mkdir(parents=True, exist_ok=True)
    eq_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    make_postfit_residuals_linear_plot(res_std, std_dir)
    make_trace_cov_pos_plot(res_std, std_dir, "trace_cov_pos.png", "Standard SRIF trace(P_pos)")
    make_trace_cov_vel_plot(res_std, std_dir, "trace_cov_vel.png", "Standard SRIF trace(P_vel)")

    make_postfit_residuals_linear_plot(res_eq, eq_dir)
    make_trace_cov_pos_plot(res_eq, eq_dir, "trace_cov_pos.png", "Eq5.10.23 SRIF trace(P_pos)")
    make_trace_cov_vel_plot(res_eq, eq_dir, "trace_cov_vel.png", "Eq5.10.23 SRIF trace(P_vel)")

    de_m = np.asarray(res_eq.state_error_meas, dtype=float) - np.asarray(res_std.state_error_meas, dtype=float)
    dP_m2 = np.asarray(res_eq.P_meas, dtype=float) - np.asarray(res_std.P_meas, dtype=float)
    dpost_m = (
        np.asarray(res_eq.postfit_resids_linear_final, dtype=float)
        - np.asarray(res_std.postfit_resids_linear_final, dtype=float)
    )

    make_state_error_difference_plot(res_std.t_meas, de_m, cmp_dir)
    make_rminus_nontriangular_plot(
        out_eq["t_meas"],
        np.asarray(out_eq["Rminus_lower_norm"], dtype=float),
        np.asarray(out_eq["Rminus_rel_lower_norm"], dtype=float),
        cmp_dir,
    )

    rms_std = rms_by_component_2d(res_std.postfit_resids_linear_final)
    rms_eq = rms_by_component_2d(res_eq.postfit_resids_linear_final)

    print("\nHW5-1c: Eq. 5.10.23 experiment (no forced triangular R-)")
    print("  Standard SRIF linear postfit RMS [m, m/s]:")
    print(f"    rho={rms_std[0]:.6e}, rhodot={rms_std[1]:.6e}")
    print("  Eq5.10.23 SRIF linear postfit RMS [m, m/s]:")
    print(f"    rho={rms_eq[0]:.6e}, rhodot={rms_eq[1]:.6e}")

    print("\nDirect differences (Eq5.10.23 - Standard):")
    print(f"  max|d state_error| = {np.nanmax(np.abs(de_m)):.6e} [m, m/s]")
    print(f"  max|dP| = {np.nanmax(np.abs(dP_m2)):.6e} [m^2, (m/s)^2]")
    print(f"  max|d linear postfit| = {np.nanmax(np.abs(dpost_m)):.6e} [m, m/s]")
    print(
        "  max relative lower-triangular content in R- = "
        f"{np.nanmax(np.asarray(out_eq['Rminus_rel_lower_norm'], dtype=float)):.6e}"
    )

    print("\nInterpretation:")
    print("  The filter should remain stable and very similar if you do full linear solves with non-triangular R-.")
    print("  The measurement QR step re-triangularizes the posterior each epoch.")
    print("  Upper-triangular R- is mainly a computational convenience for cheap back-substitution.")
    print("  If code still assumes triangular R- (e.g., solve_triangular on non-triangular R-), behavior degrades.")

    print(f"\nSaved standard SRIF plots to: {std_dir}")
    print(f"Saved Eq5.10.23 SRIF plots to: {eq_dir}")
    print(f"Saved comparison plots to: {cmp_dir}")


if __name__ == "__main__":
    main()
