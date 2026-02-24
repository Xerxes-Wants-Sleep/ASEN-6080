import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../../")

from src.Functions.batch import batch_estimate_x0
from src.Functions.filters import LinearizedKalmanFilter
from src.Functions.postprocess_batch_for_plots import postprocess_batch_for_plots
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


def run_lkf_smoothed(*, all_meas, stations, Xtrue_meas, x0_bar, P0, R, mu, J2, J3):
    Q_zero = np.zeros((6, 6), dtype=float)

    lkf = LinearizedKalmanFilter(
        X0_star=x0_bar,
        P0=P0,
        R=R,
        Q=Q_zero,
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
        out = lkf.run(
            all_meas=all_meas,
            stations=stations,
            Xtrue_meas=Xtrue_meas,
            run_smoother=True,
            smooth_back_points=None,
        )

    if not out.get("smoother_ran", False):
        raise RuntimeError("LKF smoother did not run as expected.")
    if out.get("Xhat_smooth", None) is None:
        raise RuntimeError("LKF smoother output 'Xhat_smooth' is missing.")

    return out


def run_batch_one_iter(*, all_meas, stations, x0_bar, P0, R, mu, J2, J3):
    x0_hat, P0_hat, info = batch_estimate_x0(
        all_meas=all_meas,
        stations=stations,
        x0_bar=x0_bar,
        P0=P0,
        R=R,
        mu=mu,
        J2=J2,
        J3=J3,
        max_iter=1,
        tol=1.0e-10,
        reltol=1.0e-10,
        abstol=1.0e-10,
    )

    out = postprocess_batch_for_plots(
        all_meas=all_meas,
        stations=stations,
        x0_hat=x0_hat,
        P0_hat=P0_hat,
        mu=mu,
        J2=J2,
        J3=J3,
        reltol=1.0e-10,
        abstol=1.0e-10,
    )
    return x0_hat, P0_hat, info, out


def make_overlay_plot(t_s, X_lkf_smooth, X_batch, outdir: Path):
    t_hr = np.asarray(t_s, dtype=float) / 3600.0
    labels = ["x [km]", "y [km]", "z [km]", "vx [km/s]", "vy [km/s]", "vz [km/s]"]

    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    axs = axs.flatten()

    for i in range(6):
        axs[i].plot(t_hr, X_lkf_smooth[:, i], ".", markersize=2, label="LKF RTS smoothed (Q=0)")
        axs[i].plot(t_hr, X_batch[:, i],  ".", markersize=2, linewidth=1.2, label="Batch (1 iter)")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[-2].set_xlabel("Time [hours]")
    axs[-1].set_xlabel("Time [hours]")
    handles, leglabels = axs[0].get_legend_handles_labels()
    fig.legend(handles, leglabels, loc="upper right")
    fig.suptitle("Prob 1c: Smoothed LKF (Q=0) vs Batch (1 Iteration) State Estimate Overlay")
    fig.tight_layout()
    fig.savefig(outdir / "prob1c_overlay_lkf_smooth_vs_batch1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_difference_plot(t_s, dX_batch_minus_lkf: np.ndarray, outdir: Path):
    t_hr = np.asarray(t_s, dtype=float) / 3600.0
    labels = [
        "dx = batch - lkf [km]",
        "dy = batch - lkf [km]",
        "dz = batch - lkf [km]",
        "dvx = batch - lkf [km/s]",
        "dvy = batch - lkf [km/s]",
        "dvz = batch - lkf [km/s]",
    ]

    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    axs = axs.flatten()

    for i in range(6):
        axs[i].plot(t_hr, dX_batch_minus_lkf[:, i],  ".", markersize=2)
        axs[i].axhline(0.0, color="k", linewidth=0.8)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[-2].set_xlabel("Time [hours]")
    axs[-1].set_xlabel("Time [hours]")
    fig.suptitle("Prob 1c: Batch (1 Iteration) - Smoothed LKF (Q=0) State Difference")
    fig.tight_layout()
    fig.savefig(outdir / "prob1c_difference_batch_minus_lkf_smooth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0_bar = load_problem2_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    out_lkf = run_lkf_smoothed(
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

    _, _, _, out_batch = run_batch_one_iter(
        all_meas=all_meas,
        stations=stations,
        x0_bar=x0_bar,
        P0=P0,
        R=R,
        mu=mu,
        J2=J2,
        J3=J3,
    )

    t_lkf = np.asarray(out_lkf["t_meas"], dtype=float)
    t_batch = np.asarray(out_batch["t_meas"], dtype=float)
    if t_lkf.shape != t_batch.shape or not np.allclose(t_lkf, t_batch, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("LKF and batch comparison times are not aligned.")

    X_lkf_smooth = np.asarray(out_lkf["Xhat_smooth"], dtype=float)
    X_batch = np.asarray(out_batch["xhat_meas"], dtype=float)
    if X_lkf_smooth.shape != X_batch.shape:
        raise RuntimeError("LKF smoothed and batch state histories do not have matching shapes.")

    dX = X_batch - X_lkf_smooth

    labels = ["x", "y", "z", "vx", "vy", "vz"]
    print("\nProb 1c comparison: Batch (1 iter) vs Smoothed LKF (Q=0)")
    for i, lab in enumerate(labels):
        print(
            f"  {lab:>3s}: max|diff| = {np.nanmax(np.abs(dX[:, i])):.6e}, "
            f"RMS diff = {rms_nan(dX[:, i]):.6e}"
        )
    print(
        f"  |dr| RMS = {rms_nan(np.linalg.norm(dX[:, 0:3], axis=1)):.6e} km, "
        f"|dv| RMS = {rms_nan(np.linalg.norm(dX[:, 3:6], axis=1)):.6e} km/s"
    )

    plot_dir = Path(__file__).resolve().parent / "Plots" / "Prob1c"
    plot_dir.mkdir(parents=True, exist_ok=True)

    make_overlay_plot(t_lkf, X_lkf_smooth, X_batch, plot_dir)
    make_difference_plot(t_lkf, dX, plot_dir)

    print(f"\nSaved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
