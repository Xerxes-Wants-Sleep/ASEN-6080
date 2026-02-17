import sys
from pathlib import Path
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../../")

from src.Functions.filters import ExtendedKalmanFilter
from src.Functions.stations import Stations


def rms_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(x**2)))


def build_q_accel_cov_km(sigma_m_s2: float) -> np.ndarray:
    sigma_km_s2 = float(sigma_m_s2) * 1.0e-3
    return np.diag([sigma_km_s2**2, sigma_km_s2**2, sigma_km_s2**2])


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
    P0 = np.diag(
        [
            sigma_r0_km**2,
            sigma_r0_km**2,
            sigma_r0_km**2,
            sigma_v0_km_s**2,
            sigma_v0_km_s**2,
            sigma_v0_km_s**2,
        ]
    )

    dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
    x0 = x0_true + dx

    return all_meas, Xtrue_meas, stations, R, P0, x0


def compute_metrics(out: dict) -> dict:
    postfit_lin = np.asarray(out["postfit_resids_linear_final"], dtype=float)
    state_err = np.asarray(out["state_error_meas"], dtype=float)

    return {
        "rho_postfit_rms_km": rms_nan(postfit_lin[:, 0]),
        "rhodot_postfit_rms_km_s": rms_nan(postfit_lin[:, 1]),
        "pos3_rms_km": rms_nan(np.linalg.norm(state_err[:, 0:3], axis=1)),
        "vel3_rms_km_s": rms_nan(np.linalg.norm(state_err[:, 3:6], axis=1)),
    }


def make_sweep_plots(summary_df: pd.DataFrame, outdir: Path, opt_sigma_m_s2: float) -> None:
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
    fig.suptitle("EKF SNC (RIC Q) Sweep: Linear Postfit RMS vs sigma")
    fig.tight_layout()
    fig.savefig(outdir / "ekf_snc_ric_sweep_postfit_rms.png", dpi=300, bbox_inches="tight")
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
    fig.suptitle("EKF SNC (RIC Q) Sweep: 3D State RMS vs sigma")
    fig.tight_layout()
    fig.savefig(outdir / "ekf_snc_ric_sweep_state_rms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_optimal_plots(out: dict, outdir: Path, R: np.ndarray, sigma_opt_m_s2: float) -> None:
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
    fig.suptitle(f"EKF Optimal SNC (RIC Q) State Errors (+/-3sigma), sigma={sigma_opt_m_s2:.3e} m/s^2")
    handles, leglabels = axs[0].get_legend_handles_labels()
    fig.legend(handles, leglabels, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "ekf_optimal_state_errors_eci_ricq.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Zoomed state-error view (t >= 5 hours)
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
        fig.suptitle(f"EKF Optimal SNC (RIC Q) State Errors (+/-3sigma), t >= 5 h, sigma={sigma_opt_m_s2:.3e} m/s^2")
        handles, leglabels = axs[0].get_legend_handles_labels()
        fig.legend(handles, leglabels, loc="upper right")
        fig.tight_layout()
        fig.savefig(outdir / "ekf_optimal_state_errors_eci_zoom_t_ge_5h_ricq.png", dpi=300, bbox_inches="tight")
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

    fig.suptitle(f"EKF Optimal SNC (RIC Q) Linear Postfit Residuals, sigma={sigma_opt_m_s2:.3e} m/s^2")
    fig.tight_layout()
    fig.savefig(outdir / "ekf_optimal_postfit_linear_ricq.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0 = load_problem2_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    sigma_sweep_m_s2 = np.logspace(-15, -2, 13)

    plot_dir = Path("Plots") / "EKF_SNC_RIC"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    out_by_sigma = {}

    for sigma_m_s2 in sigma_sweep_m_s2:
        Q_accel = build_q_accel_cov_km(sigma_m_s2)
        ekf = ExtendedKalmanFilter(
            x0=x0,
            P0=P0,
            R=R,
            Q=Q_accel,
            mu=mu,
            J2=J2,
            J3=J3,
            reltol=1e-10,
            abstol=1e-10,
            method="DOP853",
            j2=True,
            j3=False,
            first_pass_gap_s=6 * 3600.0,
        )
        ekf.q_frame = "ric"

        run_error = None
        out = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = ekf.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"

        if out is not None:
            metrics = compute_metrics(out)
        else:
            metrics = {
                "rho_postfit_rms_km": np.nan,
                "rhodot_postfit_rms_km_s": np.nan,
                "pos3_rms_km": np.nan,
                "vel3_rms_km_s": np.nan,
            }

        row = {"sigma_m_s2": sigma_m_s2}
        row.update(metrics)
        row["run_ok"] = out is not None
        row["run_error"] = run_error
        rows.append(row)
        if out is not None:
            out_by_sigma[sigma_m_s2] = out

    summary_df = pd.DataFrame(rows).sort_values("sigma_m_s2")
    valid_df = summary_df[np.isfinite(summary_df["pos3_rms_km"].to_numpy(float))]
    if valid_df.empty:
        summary_df.to_csv(plot_dir / "ekf_snc_ric_sweep_summary.csv", index=False)
        raise RuntimeError("All EKF SNC RIC sigma runs failed; no valid point available for optimal-case plots.")

    best_idx = valid_df["pos3_rms_km"].to_numpy(float).argmin()
    sigma_opt_m_s2 = float(valid_df.iloc[best_idx]["sigma_m_s2"])

    summary_df.to_csv(plot_dir / "ekf_snc_ric_sweep_summary.csv", index=False)
    make_sweep_plots(valid_df, plot_dir, sigma_opt_m_s2)

    out_opt = out_by_sigma[sigma_opt_m_s2]

    opt_dir = plot_dir / "Optimal"
    opt_dir.mkdir(parents=True, exist_ok=True)

    make_optimal_plots(out_opt, opt_dir, R, sigma_opt_m_s2)

    print(f"\nOptimal sigma (EKF RIC-Q, by minimum 3D position RMS): {sigma_opt_m_s2:.6e} m/s^2")
    print(valid_df.iloc[best_idx].to_string())
    failed_count = int((~summary_df["run_ok"]).sum())
    print(f"Failed sigma runs: {failed_count} / {len(summary_df)}")
    print(f"\nSaved SNC sweep + optimal-case plots to: {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
