import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../../")

from src.Functions.filters import LinearizedKalmanFilter
from src.Functions.srif import SquareRootInformationFilter
from src.Functions.stations import Stations
from src.helpers.plotting.common import as_hours, savefig
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.post_processing import run_filter_post_processing


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

    dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)/4
    x0_bar = x0_true + dx

    return all_meas, Xtrue_meas, stations, R, P0, x0_bar


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


def make_trace_cov_pos_plot(result, outdir: Path, filename: str, title: str):
    t_hr = as_hours(result.t_meas)
    P = np.asarray(result.P_meas, dtype=float)
    tr_pos = np.sum(np.maximum(np.diagonal(P[:, 0:3, 0:3], axis1=1, axis2=2), 0.0), axis=1)

    fig = plt.figure()
    plt.semilogy(t_hr, tr_pos, ".", markersize=2)
    plt.xlabel("Time [hours]")
    plt.ylabel("trace(P_pos) [m^2]")
    plt.title(title)
    savefig(fig, outdir / filename)


def make_trace_cov_vel_plot(result, outdir: Path, filename: str, title: str):
    t_hr = as_hours(result.t_meas)
    P = np.asarray(result.P_meas, dtype=float)
    tr_vel = np.sum(np.maximum(np.diagonal(P[:, 3:6, 3:6], axis1=1, axis2=2), 0.0), axis=1)

    fig = plt.figure()
    plt.semilogy(t_hr, tr_vel, ".", markersize=2)
    plt.xlabel("Time [hours]")
    plt.ylabel("trace(P_vel) [(m/s)^2]")
    plt.title(title)
    savefig(fig, outdir / filename)


def rms_by_component_2d(x: np.ndarray):
    return np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2, axis=0))


def run_lkf(*, all_meas, stations, Xtrue_meas, x0_bar, P0, R, mu, J2, J3):
    lkf = LinearizedKalmanFilter(
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
    )

    with contextlib.redirect_stdout(io.StringIO()):
        return lkf.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)


def run_srif(*, all_meas, stations, Xtrue_meas, x0_bar, P0, R, mu, J2, J3):
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


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0_bar = load_problem2_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    out_lkf = run_lkf(
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
    out_srif = run_srif(
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

    result_lkf = convert_filter_output_km_to_m_for_plotting(out_lkf, R_km=R)
    result_srif = convert_filter_output_km_to_m_for_plotting(out_srif, R_km=R)

    plot_root = Path(__file__).resolve().parent / "Plots"
    lkf_dir = plot_root / "LKF"
    srif_dir = plot_root / "SRIF"
    lkf_dir.mkdir(parents=True, exist_ok=True)
    srif_dir.mkdir(parents=True, exist_ok=True)

    make_postfit_residuals_linear_plot(result_lkf, lkf_dir)
    make_trace_cov_pos_plot(result_lkf, lkf_dir, "trace_cov_pos.png", "LKF trace(P_pos)")
    make_trace_cov_vel_plot(result_lkf, lkf_dir, "trace_cov_vel.png", "LKF trace(P_vel)")

    make_postfit_residuals_linear_plot(result_srif, srif_dir)
    make_trace_cov_pos_plot(result_srif, srif_dir, "trace_cov_pos.png", "SRIF trace(P_pos)")
    make_trace_cov_vel_plot(result_srif, srif_dir, "trace_cov_vel.png", "SRIF trace(P_vel)")

    rms_lkf = rms_by_component_2d(result_lkf.postfit_resids_linear_final)
    rms_srif = rms_by_component_2d(result_srif.postfit_resids_linear_final)

    print("\nSRIF pre-whitening note:")
    print("  SRIF.run() already pre-whitens residuals and H internally using Cholesky(R).")
    print("  No extra pre-whitening is needed in this run script.")

    print("\nLinearized postfit RMS comparison (meters, meters/sec):")
    print(f"  LKF  : rho={rms_lkf[0]:.6e}, rhodot={rms_lkf[1]:.6e}")
    print(f"  SRIF : rho={rms_srif[0]:.6e}, rhodot={rms_srif[1]:.6e}")

    print(f"\nSaved LKF plots to:  {lkf_dir}")
    print(f"Saved SRIF plots to: {srif_dir}")


if __name__ == "__main__":
    main()
