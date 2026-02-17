import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../../")

from src.Functions.dmc import lkf_with_dmc
from src.Functions.jacobians import accel_wJ2J3
from src.Functions.stations import Stations


def rms_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(x**2)))


def build_stations():
    theta0_deg = 122.0
    w_earth_rad_per_s = 2.0 * np.pi / (3600.0 * 24.0)
    radius_earth_km = 6378.0

    return [
        Stations(
            "Station 1",
            lat_deg=-35.398333,
            lon_deg=148.981944,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "Station 2",
            lat_deg=40.427222,
            lon_deg=355.749444,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "Station 3",
            lat_deg=35.247164,
            lon_deg=243.205000,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
    ]


def load_problem2_inputs():
    df_meas = pd.read_csv("../Homework2/meas_data/prob2_hw2_measurements_noisy.csv")
    all_meas = sorted(df_meas.to_dict(orient="records"), key=lambda m: float(m["t"]))
    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)

    truth_df = pd.read_csv("../Homework1/HW2_j3_on_truth.csv")
    truth_times = truth_df["t_s"].to_numpy(float)
    truth_states = truth_df[["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
    x0_true = truth_states[0, :]
    Xtrue_meas = np.column_stack([np.interp(t_meas, truth_times, truth_states[:, k]) for k in range(6)])

    stations = build_stations()

    sigma_rho_km = 1.0e-3
    sigma_rhod_km_s = 1.0e-6
    R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    sigma_r0_km = 1.0
    sigma_v0_km_s = 1.0e-3
    sigma_w0_km_s2 = 1.0e-9
    P0 = np.diag(
        [
            sigma_r0_km**2,
            sigma_r0_km**2,
            sigma_r0_km**2,
            sigma_v0_km_s**2,
            sigma_v0_km_s**2,
            sigma_v0_km_s**2,
            sigma_w0_km_s2**2,
            sigma_w0_km_s2**2,
            sigma_w0_km_s2**2,
        ]
    )

    dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
    x0_star = x0_true + dx

    return all_meas, Xtrue_meas, stations, R, P0, x0_star


def initial_orbit_period_s(x0_6: np.ndarray, mu_km3_s2: float) -> float:
    r = np.asarray(x0_6[:3], dtype=float)
    v = np.asarray(x0_6[3:6], dtype=float)
    rmag = float(np.linalg.norm(r))
    vmag = float(np.linalg.norm(v))
    eps = 0.5 * vmag * vmag - float(mu_km3_s2) / rmag
    a = -float(mu_km3_s2) / (2.0 * eps)
    return float(2.0 * np.pi * np.sqrt(a**3 / float(mu_km3_s2)))


def compute_w_ref_from_truth(Xtrue_meas: np.ndarray, mu: float, J2: float, J3: float, Re: float = 6378.0) -> np.ndarray:
    Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
    w_ref = np.zeros((Xtrue_meas.shape[0], 3), dtype=float)
    for k in range(Xtrue_meas.shape[0]):
        r = Xtrue_meas[k, 0:3]
        a_truth = accel_wJ2J3(r, mu, J2, J3, Re=Re, j2=True, j3=True)
        a_filter = accel_wJ2J3(r, mu, J2, 0.0, Re=Re, j2=True, j3=False)
        w_ref[k, :] = a_truth - a_filter
    return w_ref


def load_w_ref_from_csv(csv_path: Path, t_target: np.ndarray) -> np.ndarray | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    time_candidates = ["t_s", "t", "time_s", "time", "Time(s)"]
    comp_candidates = [
        ["wx_km_s2", "wy_km_s2", "wz_km_s2"],
        ["w_x_km_s2", "w_y_km_s2", "w_z_km_s2"],
        ["wx", "wy", "wz"],
        ["w_x", "w_y", "w_z"],
        ["ax_km_s2", "ay_km_s2", "az_km_s2"],
        ["ax", "ay", "az"],
    ]

    t_col = next((c for c in time_candidates if c in df.columns), None)
    if t_col is None:
        return None
    comp_cols = next((cols for cols in comp_candidates if all(c in df.columns for c in cols)), None)
    if comp_cols is None:
        return None

    t_src = df[t_col].to_numpy(float)
    w_src = df[comp_cols].to_numpy(float)
    if len(t_src) < 2:
        return None

    w_ref = np.column_stack([np.interp(t_target, t_src, w_src[:, i]) for i in range(3)])
    return np.asarray(w_ref, dtype=float)


def compute_metrics(out: dict) -> dict:
    postfit_lin = np.asarray(out["postfit_resids_linear_final"], dtype=float)
    state_err = np.asarray(out["state_error_meas"], dtype=float)
    return {
        "rho_postfit_rms_km": rms_nan(postfit_lin[:, 0]),
        "rhodot_postfit_rms_km_s": rms_nan(postfit_lin[:, 1]),
        "pos3_rms_km": rms_nan(np.linalg.norm(state_err[:, 0:3], axis=1)),
        "vel3_rms_km_s": rms_nan(np.linalg.norm(state_err[:, 3:6], axis=1)),
    }


def make_sweep_plots(summary_df: pd.DataFrame, outdir: Path, opt_sigma_m_s2: float, tau_s: float) -> None:
    sigma = summary_df["sigma_m_s2"].to_numpy(float)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    axs[0].loglog(sigma, summary_df["rho_postfit_rms_km"], "o-", markersize=4)
    axs[0].axvline(opt_sigma_m_s2, color="k", linestyle="--", linewidth=1)
    axs[0].set_ylabel("RMS range postfit (linear) [km]")
    axs[0].grid(True, which="both")

    axs[1].loglog(sigma, summary_df["rhodot_postfit_rms_km_s"], "o-", markersize=4)
    axs[1].axvline(opt_sigma_m_s2, color="k", linestyle="--", linewidth=1, label="optimal sigma")
    axs[1].set_ylabel("RMS range-rate postfit (linear) [km/s]")
    axs[1].set_xlabel("sigma [m/s^2]")
    axs[1].grid(True, which="both")
    axs[1].legend(loc="best")
    fig.suptitle(f"LKF DMC Sweep (tau={tau_s:.1f}s): Linear Postfit RMS vs sigma")
    fig.tight_layout()
    fig.savefig(outdir / "lkf_dmc_sweep_postfit_rms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    axs[0].loglog(sigma, summary_df["pos3_rms_km"], "o-", markersize=4)
    axs[0].axvline(opt_sigma_m_s2, color="k", linestyle="--", linewidth=1)
    axs[0].set_ylabel("3D position RMS [km]")
    axs[0].grid(True, which="both")

    axs[1].loglog(sigma, summary_df["vel3_rms_km_s"], "o-", markersize=4)
    axs[1].axvline(opt_sigma_m_s2, color="k", linestyle="--", linewidth=1, label="optimal sigma")
    axs[1].set_ylabel("3D velocity RMS [km/s]")
    axs[1].set_xlabel("sigma [m/s^2]")
    axs[1].grid(True, which="both")
    axs[1].legend(loc="best")
    fig.suptitle(f"LKF DMC Sweep (tau={tau_s:.1f}s): 3D State RMS vs sigma")
    fig.tight_layout()
    fig.savefig(outdir / "lkf_dmc_sweep_state_rms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_optimal_plots(
    out: dict,
    outdir: Path,
    R: np.ndarray,
    sigma_opt_m_s2: float,
    tau_s: float,
    w_ref_km_s2: np.ndarray | None = None,
) -> None:
    t_hr = np.asarray(out["t_meas"], dtype=float) / 3600.0
    err = np.asarray(out["state_error_meas"], dtype=float)
    P = np.asarray(out["P_meas"], dtype=float)
    sig3 = 3.0 * np.sqrt(np.maximum(np.diagonal(P, axis1=1, axis2=2), 0.0))

    labels = ["x [km]", "y [km]", "z [km]", "vx [km/s]", "vy [km/s]", "vz [km/s]"]
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axs = axs.flatten()
    for i in range(6):
        axs[i].plot(t_hr, err[:, i], ".", markersize=2, label="error")
        axs[i].plot(t_hr, sig3[:, i], "r", linewidth=1.0, label="+3sigma")
        axs[i].plot(t_hr, -sig3[:, i], "r", linewidth=1.0, label="-3sigma")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
    axs[-2].set_xlabel("Time [hours]")
    axs[-1].set_xlabel("Time [hours]")
    fig.suptitle(f"LKF DMC Optimal State Errors (+/-3sigma), sigma={sigma_opt_m_s2:.3e} m/s^2, tau={tau_s:.1f}s")
    handles, leglabels = axs[0].get_legend_handles_labels()
    fig.legend(handles, leglabels, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "lkf_dmc_optimal_state_errors_eci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    mask = t_hr >= 5.0
    if np.any(mask):
        t_hr_z = t_hr[mask]
        err_z = err[mask, :]
        sig3_z = sig3[mask, :]

        fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        axs = axs.flatten()
        for i in range(6):
            axs[i].plot(t_hr_z, err_z[:, i], ".", markersize=2, label="error")
            axs[i].plot(t_hr_z, sig3_z[:, i], "r", linewidth=1.0, label="+3sigma")
            axs[i].plot(t_hr_z, -sig3_z[:, i], "r", linewidth=1.0, label="-3sigma")
            axs[i].set_ylabel(labels[i])
            axs[i].grid(True)
        axs[-2].set_xlabel("Time [hours]")
        axs[-1].set_xlabel("Time [hours]")
        fig.suptitle(f"LKF DMC Optimal State Errors (+/-3sigma), t >= 5 h, sigma={sigma_opt_m_s2:.3e} m/s^2")
        handles, leglabels = axs[0].get_legend_handles_labels()
        fig.legend(handles, leglabels, loc="upper right")
        fig.tight_layout()
        fig.savefig(outdir / "lkf_dmc_optimal_state_errors_eci_zoom_t_ge_5h.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    pf_lin = np.asarray(out["postfit_resids_linear_final"], dtype=float)
    sigma_rho = float(np.sqrt(R[0, 0]))
    sigma_rhodot = float(np.sqrt(R[1, 1]))

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(t_hr, pf_lin[:, 0], ".", markersize=2)
    axs[0].axhline(0.0, color="k", linewidth=1.0)
    axs[0].axhline(+3.0 * sigma_rho, color="k", linestyle="--", linewidth=1.0)
    axs[0].axhline(-3.0 * sigma_rho, color="k", linestyle="--", linewidth=1.0)
    axs[0].set_ylabel("Linear postfit rho [km]")
    axs[0].grid(True)

    axs[1].plot(t_hr, pf_lin[:, 1], ".", markersize=2)
    axs[1].axhline(0.0, color="k", linewidth=1.0)
    axs[1].axhline(+3.0 * sigma_rhodot, color="k", linestyle="--", linewidth=1.0)
    axs[1].axhline(-3.0 * sigma_rhodot, color="k", linestyle="--", linewidth=1.0)
    axs[1].set_ylabel("Linear postfit rhodot [km/s]")
    axs[1].set_xlabel("Time [hours]")
    axs[1].grid(True)

    fig.suptitle(f"LKF DMC Optimal Linear Postfit Residuals, sigma={sigma_opt_m_s2:.3e} m/s^2")
    fig.tight_layout()
    fig.savefig(outdir / "lkf_dmc_optimal_postfit_linear.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # DMC diagnostics: w estimate with covariance and interval Q_ww(t).
    diag_dir = outdir / "Diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    X9 = np.asarray(out.get("Xhat_meas_9", np.empty((0, 9))), dtype=float)
    P9 = np.asarray(out.get("P_meas_9", np.empty((0, 9, 9))), dtype=float)
    if X9.ndim == 2 and X9.shape[1] >= 9 and P9.ndim == 3 and P9.shape[1] >= 9:
        w_hat = X9[:, 6:9]
        w_sig3 = 3.0 * np.sqrt(np.maximum(np.diagonal(P9[:, 6:9, 6:9], axis1=1, axis2=2), 0.0))
        w_labels = ["w_x", "w_y", "w_z"]

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            axs[i].plot(t_hr, w_hat[:, i], ".", markersize=2, label="w_hat")
            if w_ref_km_s2 is not None and np.asarray(w_ref_km_s2).shape == w_hat.shape:
                axs[i].plot(t_hr, w_ref_km_s2[:, i], "k", linewidth=1.0, label="J3 accel ref")
            axs[i].plot(t_hr, w_sig3[:, i], "r", linewidth=1.0, label="+3sigma")
            axs[i].plot(t_hr, -w_sig3[:, i], "r", linewidth=1.0, label="-3sigma")
            axs[i].set_ylabel(f"{w_labels[i]} [km/s^2]")
            axs[i].grid(True)
        axs[-1].set_xlabel("Time [hours]")
        fig.suptitle(f"LKF DMC Optimal w Estimate (+/-3sigma), sigma={sigma_opt_m_s2:.3e} m/s^2")
        handles, leglabels = axs[0].get_legend_handles_labels()
        fig.legend(handles, leglabels, loc="upper right")
        fig.tight_layout()
        fig.savefig(diag_dir / "lkf_dmc_optimal_w_estimate.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        mask_w = t_hr >= 4.0
        if np.any(mask_w):
            t_hr_w = t_hr[mask_w]
            w_hat_w = w_hat[mask_w, :]
            w_sig3_w = w_sig3[mask_w, :]
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for i in range(3):
                axs[i].plot(t_hr_w, w_hat_w[:, i], ".", markersize=2, label="w_hat")
                if w_ref_km_s2 is not None and np.asarray(w_ref_km_s2).shape == w_hat.shape:
                    axs[i].plot(t_hr_w, w_ref_km_s2[mask_w, i], "k", linewidth=1.0, label="J3 accel ref")
                axs[i].plot(t_hr_w, w_sig3_w[:, i], "r", linewidth=1.0, label="+3sigma")
                axs[i].plot(t_hr_w, -w_sig3_w[:, i], "r", linewidth=1.0, label="-3sigma")
                axs[i].set_ylabel(f"{w_labels[i]} [km/s^2]")
                axs[i].grid(True)
            axs[-1].set_xlabel("Time [hours]")
            fig.suptitle(
                f"LKF DMC Optimal w Estimate (+/-3sigma), t >= 4 h, sigma={sigma_opt_m_s2:.3e} m/s^2"
            )
            handles, leglabels = axs[0].get_legend_handles_labels()
            fig.legend(handles, leglabels, loc="upper right")
            fig.tight_layout()
            fig.savefig(diag_dir / "lkf_dmc_optimal_w_estimate_t_ge_4h.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    qww = np.asarray(out.get("Qk_interval_ww_diag", np.empty((0, 3))), dtype=float)
    if qww.ndim == 2 and qww.shape[1] == 3 and qww.shape[0] == t_hr.shape[0]:
        q_labels = ["Qww_xx", "Qww_yy", "Qww_zz"]
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            axs[i].semilogy(t_hr, np.maximum(qww[:, i], 1.0e-30), ".", markersize=2)
            axs[i].set_ylabel(f"{q_labels[i]} [km^2/s^4]")
            axs[i].grid(True, which="both")
        axs[-1].set_xlabel("Time [hours]")
        fig.suptitle(f"LKF DMC Optimal Interval Qww vs Time, sigma={sigma_opt_m_s2:.3e} m/s^2")
        fig.tight_layout()
        fig.savefig(diag_dir / "lkf_dmc_optimal_qww_vs_time.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if P9.ndim == 3 and P9.shape[1] >= 9 and P9.shape[0] == t_hr.shape[0]:
        pww = np.maximum(np.diagonal(P9[:, 6:9, 6:9], axis1=1, axis2=2), 1.0e-30)
        p_labels = ["Pww_xx", "Pww_yy", "Pww_zz"]
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            axs[i].semilogy(t_hr, pww[:, i], ".", markersize=2)
            axs[i].set_ylabel(f"{p_labels[i]} [km^2/s^4]")
            axs[i].grid(True, which="both")
        axs[-1].set_xlabel("Time [hours]")
        fig.suptitle(f"LKF DMC Optimal Pww vs Time, sigma={sigma_opt_m_s2:.3e} m/s^2")
        fig.tight_layout()
        fig.savefig(diag_dir / "lkf_dmc_optimal_pww_vs_time.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0_star = load_problem2_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    period_s = initial_orbit_period_s(x0_star, mu)
    tau_s = period_s / 30.0

    sigma_sweep_m_s2 = np.logspace(-15, -2, 13)

    plot_dir = Path("Plots") / "LKF_DMC"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    out_by_sigma = {}

    for sigma_m_s2 in sigma_sweep_m_s2:
        run_error = None
        out = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = lkf_with_dmc(
                    all_meas=all_meas,
                    stations=stations,
                    X0_star=x0_star,
                    P0=P0,
                    R=R,
                    mu=mu,
                    J2=J2,
                    J3=J3,
                    tau_s=tau_s,
                    sigma_accel_m_s2=float(sigma_m_s2),
                    Xtrue_meas=Xtrue_meas,
                    reltol=1.0e-10,
                    abstol=1.0e-10,
                    method="DOP853",
                    j2=True,
                    j3=False,
                    first_pass_gap_s=6 * 3600.0,
                )
        except RuntimeError as exc:
            run_error = str(exc)

        if out is not None:
            metrics = compute_metrics(out)
        else:
            metrics = {
                "rho_postfit_rms_km": np.nan,
                "rhodot_postfit_rms_km_s": np.nan,
                "pos3_rms_km": np.nan,
                "vel3_rms_km_s": np.nan,
            }

        row = {"sigma_m_s2": float(sigma_m_s2)}
        row.update(metrics)
        row["run_ok"] = out is not None
        row["run_error"] = run_error
        rows.append(row)

        if out is not None:
            out_by_sigma[float(sigma_m_s2)] = out

    summary_df = pd.DataFrame(rows).sort_values("sigma_m_s2")
    valid_df = summary_df[np.isfinite(summary_df["pos3_rms_km"].to_numpy(float))]
    summary_df.to_csv(plot_dir / "lkf_dmc_sweep_summary.csv", index=False)

    if valid_df.empty:
        raise RuntimeError("All LKF DMC sigma runs failed; no valid point available for optimal-case plots.")

    best_idx = valid_df["pos3_rms_km"].to_numpy(float).argmin()
    sigma_opt_m_s2 = float(valid_df.iloc[best_idx]["sigma_m_s2"])

    make_sweep_plots(valid_df, plot_dir, sigma_opt_m_s2, tau_s)

    out_opt = out_by_sigma[sigma_opt_m_s2]
    opt_dir = plot_dir / "Optimal"
    opt_dir.mkdir(parents=True, exist_ok=True)
    t_opt = np.asarray(out_opt["t_meas"], dtype=float)

    # Prefer user-saved J3 acceleration CSV if present; otherwise compute from truth states.
    w_ref = None
    csv_candidates = [
        Path("../Homework2/j3_accel_reference.csv"),
        Path("../Homework2/j3_accel.csv"),
        Path("../Homework2/meas_data/j3_accel_reference.csv"),
        Path("../Homework2/meas_data/prob2_j3_accel.csv"),
    ]
    for cpath in csv_candidates:
        w_ref = load_w_ref_from_csv(cpath, t_opt)
        if w_ref is not None:
            print(f"Using J3 acceleration overlay CSV: {cpath}")
            break
    if w_ref is None:
        w_ref_all = compute_w_ref_from_truth(Xtrue_meas, mu=mu, J2=J2, J3=J3)
        if w_ref_all.shape[0] == t_opt.shape[0]:
            w_ref = w_ref_all
        else:
            # Safety: if lengths mismatch, interpolate from measurement times.
            t_all = np.array([float(m["t"]) for m in all_meas], dtype=float)
            w_ref = np.column_stack([np.interp(t_opt, t_all, w_ref_all[:, i]) for i in range(3)])
        print("J3 acceleration overlay CSV not found; using model-difference reference from truth states.")

    make_optimal_plots(out_opt, opt_dir, R, sigma_opt_m_s2, tau_s, w_ref_km_s2=w_ref)

    print(f"Initial orbit period used for tau: P = {period_s:.3f} s")
    print(f"DMC tau = P/30 = {tau_s:.3f} s")
    print(f"\nOptimal sigma (LKF DMC, by minimum 3D position RMS): {sigma_opt_m_s2:.6e} m/s^2")
    print(valid_df.iloc[best_idx].to_string())
    failed_count = int((~summary_df["run_ok"]).sum())
    print(f"Failed sigma runs: {failed_count} / {len(summary_df)}")
    print(f"\nSaved DMC sweep + optimal-case plots to: {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
