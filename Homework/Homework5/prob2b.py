import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../")

from prob1b import (
    convert_filter_output_km_to_m_for_plotting,
    load_problem2_inputs,
    make_trace_cov_pos_plot,
    make_trace_cov_vel_plot,
    rms_by_component_2d,
)
from src.Functions.filters import LinearizedKalmanFilter
from src.Functions.srif import SquareRootInformationFilter
from src.helpers.plotting.common import as_hours, savefig
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot


def build_q_accel_cov_km(sigma_m_s2: float) -> np.ndarray:
    sigma_km_s2 = float(sigma_m_s2) * 1.0e-3
    return np.diag([sigma_km_s2**2, sigma_km_s2**2, sigma_km_s2**2])


def run_ckf_snc(*, all_meas, stations, Xtrue_meas, x0_bar, P0, R, mu, J2, J3, sigma_m_s2: float):
    # CKF/LKF with SNC via 3x3 continuous acceleration covariance.
    ckf = LinearizedKalmanFilter(
        X0_star=x0_bar,
        P0=P0,
        R=R,
        Q=build_q_accel_cov_km(sigma_m_s2),
        mu=mu,
        J2=J2,
        J3=J3,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
        j2=True,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return ckf.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)


def run_srif_snc(*, all_meas, stations, Xtrue_meas, x0_bar, P0, R, mu, J2, J3, sigma_m_s2: float):
    # SRIF SNC in low-rank form (q=3 acceleration noise).
    # Using process_noise_mode='accel' avoids factoring singular 6x6 Qk.
    srif = SquareRootInformationFilter(
        X0_star=x0_bar,
        P0=P0,
        R=R,
        Q=build_q_accel_cov_km(sigma_m_s2),
        mu=mu,
        J2=J2,
        J3=J3,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
        j2=True,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
        process_noise_mode="accel",
        uBar=np.zeros(3, dtype=float),
        max_process_dt_s=np.inf,
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
    fig.suptitle("State-Error Difference: SRIF - CKF(SNC)")
    savefig(fig, outdir / "state_error_difference_srif_minus_ckf_snc.png")


def make_linear_postfit_difference_plot(t_s: np.ndarray, dpost_m: np.ndarray, outdir: Path):
    t_hr = as_hours(t_s)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(t_hr, dpost_m[:, 0], ".", markersize=2)
    axs[0].axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
    axs[0].set_ylabel("d rho [m]")
    axs[0].set_title("Linearized Postfit Difference: SRIF - CKF(SNC)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_hr, dpost_m[:, 1], ".", markersize=2)
    axs[1].axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
    axs[1].set_ylabel("d rhodot [m/s]")
    axs[1].set_xlabel("Time [hours]")
    axs[1].grid(True, alpha=0.3)

    savefig(fig, outdir / "linear_postfit_difference_srif_minus_ckf_snc.png")


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0_bar = load_problem2_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6
    sigma_m_s2 = 1.668e-8

    out_ckf = run_ckf_snc(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0_bar=x0_bar,
        P0=P0,
        R=R,
        mu=mu,
        J2=J2,
        J3=J3,
        sigma_m_s2=sigma_m_s2,
    )
    out_srif = run_srif_snc(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0_bar=x0_bar,
        P0=P0,
        R=R,
        mu=mu,
        J2=J2,
        J3=J3,
        sigma_m_s2=sigma_m_s2,
    )

    res_ckf = convert_filter_output_km_to_m_for_plotting(out_ckf, R_km=R)
    res_srif = convert_filter_output_km_to_m_for_plotting(out_srif, R_km=R)

    plot_root = Path(__file__).resolve().parent / "Plots" / "prob2b"
    ckf_dir = plot_root / "CKF_SNC"
    srif_dir = plot_root / "SRIF"
    cmp_dir = plot_root / "Comparison"
    ckf_dir.mkdir(parents=True, exist_ok=True)
    srif_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    make_postfit_residuals_linear_plot(res_ckf, ckf_dir)
    make_trace_cov_pos_plot(res_ckf, ckf_dir, "trace_cov_pos.png", "CKF(SNC) trace(P_pos)")
    make_trace_cov_vel_plot(res_ckf, ckf_dir, "trace_cov_vel.png", "CKF(SNC) trace(P_vel)")

    make_postfit_residuals_linear_plot(res_srif, srif_dir)
    make_trace_cov_pos_plot(res_srif, srif_dir, "trace_cov_pos.png", "SRIF(SNC) trace(P_pos)")
    make_trace_cov_vel_plot(res_srif, srif_dir, "trace_cov_vel.png", "SRIF(SNC) trace(P_vel)")

    de_m = np.asarray(res_srif.state_error_meas, dtype=float) - np.asarray(res_ckf.state_error_meas, dtype=float)
    dpost_m = (
        np.asarray(res_srif.postfit_resids_linear_final, dtype=float)
        - np.asarray(res_ckf.postfit_resids_linear_final, dtype=float)
    )
    make_state_error_difference_plot(res_ckf.t_meas, de_m, cmp_dir)
    make_linear_postfit_difference_plot(res_ckf.t_meas, dpost_m, cmp_dir)

    rms_ckf = rms_by_component_2d(res_ckf.postfit_resids_linear_final)
    rms_srif = rms_by_component_2d(res_srif.postfit_resids_linear_final)

    print("\nProb 2b: SRIF vs CKF(SNC) on HW2 data")
    print(f"  sigma used to build Q_accel: {sigma_m_s2:.6e} m/s^2")
    print("  SRIF pre-whitening is done internally in SRIF.run().")
    print("\nLinearized postfit RMS [m, m/s]:")
    print(f"  CKF(SNC): rho={rms_ckf[0]:.6e}, rhodot={rms_ckf[1]:.6e}")
    print(f"  SRIF(SNC): rho={rms_srif[0]:.6e}, rhodot={rms_srif[1]:.6e}")
    print("\nDirect differences (SRIF - CKF):")
    print(f"  max|d state_error| = {np.nanmax(np.abs(de_m)):.6e} [m, m/s]")
    print(f"  max|d linear postfit| = {np.nanmax(np.abs(dpost_m)):.6e} [m, m/s]")

    print(f"\nSaved CKF plots to: {ckf_dir}")
    print(f"Saved SRIF plots to: {srif_dir}")
    print(f"Saved comparison plots to: {cmp_dir}")


if __name__ == "__main__":
    main()
