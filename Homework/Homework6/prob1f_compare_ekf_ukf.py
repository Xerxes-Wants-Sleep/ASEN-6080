import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("../../")

from src.Functions.filters import ExtendedKalmanFilter, UnscentedKalmanFilter
from src.Functions.stations import Stations
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_postfit_residuals_nonlinear import make_postfit_residuals_nonlinear_plot
from src.helpers.plotting.plot_state_errors_eci import make_state_errors_eci_plots
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.post_processing import run_filter_post_processing


# Edit this as needed (km, km/s):
DX_INIT = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)*100


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


def load_common_inputs():
    script_dir = Path(__file__).resolve().parent

    meas_path = script_dir.parent / "Homework2" / "meas_data" / "prob2_hw2_measurements_noisy.csv"
    truth_path = script_dir.parent / "Homework1" / "HW2_j3_on_truth.csv"

    df_meas = pd.read_csv(meas_path)
    all_meas = sorted(df_meas.to_dict(orient="records"), key=lambda m: float(m["t"]))
    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)

    truth_df = pd.read_csv(truth_path)
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

    x0 = x0_true + DX_INIT
    return all_meas, Xtrue_meas, stations, R, P0, x0


def build_q_accel_cov_km(sigma_m_s2: float) -> np.ndarray:
    sigma_km_s2 = float(sigma_m_s2) * 1.0e-3
    return np.diag([sigma_km_s2**2, sigma_km_s2**2, sigma_km_s2**2])


def convert_filter_output_km_to_m_for_plotting(out: dict, *, R_km: np.ndarray):
    d = dict(out)

    if "xhat_meas" not in d and "Xhat_meas" in d:
        d["xhat_meas"] = d["Xhat_meas"]
    if "P_meas" not in d and "Phat_meas" in d:
        d["P_meas"] = d["Phat_meas"]

    d["xhat_meas"] = np.asarray(d["xhat_meas"], dtype=float) * 1000.0
    d["Xhat_meas"] = np.asarray(d["Xhat_meas"], dtype=float) * 1000.0

    if d.get("X_pf", None) is not None:
        d["X_pf"] = np.asarray(d["X_pf"], dtype=float) * 1000.0
    if d.get("state_error_meas", None) is not None:
        d["state_error_meas"] = np.asarray(d["state_error_meas"], dtype=float) * 1000.0

    d["P_meas"] = np.asarray(d["P_meas"], dtype=float) * (1000.0**2)
    if d.get("Phat_meas", None) is not None:
        d["Phat_meas"] = np.asarray(d["Phat_meas"], dtype=float) * (1000.0**2)
    if d.get("P_pf", None) is not None:
        d["P_pf"] = np.asarray(d["P_pf"], dtype=float) * (1000.0**2)

    if d.get("two_sigma_meas", None) is not None:
        d["two_sigma_meas"] = np.asarray(d["two_sigma_meas"], dtype=float) * 1000.0

    for key in ("prefit_resids_final", "postfit_resids_linear_final", "postfit_resids_meas"):
        if d.get(key, None) is not None:
            d[key] = np.asarray(d[key], dtype=float) * 1000.0

    d["R"] = np.asarray(R_km, dtype=float) * (1000.0**2)
    return run_filter_post_processing(out=d)


def run_ukf(*, all_meas, stations, Xtrue_meas, x0, P0, R, Q, mu, J2, J3):
    ukf = UnscentedKalmanFilter(
        X0=x0,
        P0=P0,
        R=R,
        Q=Q,
        mu=mu,
        J2=J2,
        J3=J3,
        Re=6378.0,
        alpha=1.0,
        beta=2.0,
        kappa=None,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
        j2=True,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        return ukf.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)


def run_ekf(*, all_meas, stations, Xtrue_meas, x0, P0, R, Q, mu, J2, J3):
    ekf = ExtendedKalmanFilter(
        x0=x0,
        P0=P0,
        R=R,
        Q=Q,
        mu=mu,
        J2=J2,
        J3=J3,
        Re=6378.0,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
        j2=True,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        return ekf.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)


def summarize_metrics(label: str, out: dict) -> dict:
    rms = out.get("rms_final", {}) if out is not None else {}
    postfit_all = np.asarray(rms.get("postfit_all", [np.nan, np.nan]), dtype=float)
    postfit_ign = np.asarray(rms.get("postfit_ignore_first", [np.nan, np.nan]), dtype=float)

    row = {
        "filter": label,
        "pos3_all_km": float(rms.get("pos3_all", np.nan)),
        "pos3_ignore_first_km": float(rms.get("pos3_ignore_first", np.nan)),
        "vel3_all_km_s": float(rms.get("vel3_all", np.nan)),
        "vel3_ignore_first_km_s": float(rms.get("vel3_ignore_first", np.nan)),
        "postfit_rho_all_km": float(postfit_all[0]),
        "postfit_rhodot_all_km_s": float(postfit_all[1]),
        "postfit_rho_ignore_first_km": float(postfit_ign[0]),
        "postfit_rhodot_ignore_first_km_s": float(postfit_ign[1]),
    }
    return row


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0 = load_common_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    sigma_q_m_s2 = 3e-7
    Q_accel = build_q_accel_cov_km(sigma_q_m_s2)

    out_ekf = run_ekf(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0=x0,
        P0=P0,
        R=R,
        Q=Q_accel,
        mu=mu,
        J2=J2,
        J3=J3,
    )
    out_ukf = run_ukf(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0=x0,
        P0=P0,
        R=R,
        Q=Q_accel,
        mu=mu,
        J2=J2,
        J3=J3,
    )

    res_ekf = convert_filter_output_km_to_m_for_plotting(out_ekf, R_km=R)
    res_ukf = convert_filter_output_km_to_m_for_plotting(out_ukf, R_km=R)

    plot_root = Path(__file__).resolve().parent / "Plots" / "prob1f_compare_ekf_ukf"
    ekf_dir = plot_root / "ekf"
    ukf_dir = plot_root / "ukf"
    for d in (plot_root, ekf_dir, ukf_dir):
        d.mkdir(parents=True, exist_ok=True)

    make_state_errors_eci_plots(res_ekf, ekf_dir, zoom_t0_s=30000.0)
    make_postfit_residuals_linear_plot(res_ekf, ekf_dir)
    make_postfit_residuals_nonlinear_plot(res_ekf, ekf_dir)
    make_trace_cov_pos_vel_plot(res_ekf, ekf_dir, title="EKF: trace(P_pos), trace(P_vel)")

    make_state_errors_eci_plots(res_ukf, ukf_dir, zoom_t0_s=30000.0)
    make_postfit_residuals_linear_plot(res_ukf, ukf_dir)
    make_postfit_residuals_nonlinear_plot(res_ukf, ukf_dir)
    make_trace_cov_pos_vel_plot(res_ukf, ukf_dir, title="UKF: trace(P_pos), trace(P_vel)")

    summary_df = pd.DataFrame(
        [
            summarize_metrics("EKF", out_ekf),
            summarize_metrics("UKF", out_ukf),
        ]
    )
    summary_path = plot_root / "ekf_vs_ukf_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"DX_INIT used (km, km/s): {DX_INIT}")
    print(f"SNC sigma used: {sigma_q_m_s2:.6e} m/s^2")
    print(f"Saved EKF plots to: {ekf_dir}")
    print(f"Saved UKF plots to: {ukf_dir}")
    print(f"Saved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
