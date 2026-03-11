import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../../")

from prob1b import build_q_accel_cov_km, convert_filter_output_km_to_m_for_plotting, load_hw3_inputs
from src.Functions.filters import ExtendedKalmanFilter, UnscentedKalmanFilter
from src.helpers.plotting.common import as_hours, savefig


def get_optimal_sigma_q_m_s2(default_sigma: float = 3.8311868495572855e-08) -> float:
    script_dir = Path(__file__).resolve().parent
    summary_path = script_dir.parent / "Homework3" / "Plots" / "EKF_SNC" / "ekf_snc_sweep_summary.csv"

    if not summary_path.exists():
        return float(default_sigma)

    df = pd.read_csv(summary_path)
    if "sigma_m_s2" not in df.columns or "pos3_rms_km" not in df.columns:
        return float(default_sigma)

    pos = pd.to_numeric(df["pos3_rms_km"], errors="coerce")
    sig = pd.to_numeric(df["sigma_m_s2"], errors="coerce")
    keep = np.isfinite(pos.to_numpy()) & np.isfinite(sig.to_numpy())
    if not np.any(keep):
        return float(default_sigma)

    i_best = int(np.argmin(pos.to_numpy()[keep]))
    sigma_best = float(sig.to_numpy()[keep][i_best])
    return sigma_best


def run_ukf_baseline(
    *,
    all_meas,
    stations,
    Xtrue_meas,
    x0,
    P0,
    R,
    Q_accel,
    mu,
    J2,
    J3,
):
    ukf = UnscentedKalmanFilter(
        X0=x0,
        P0=P0,
        R=R,
        Q=Q_accel,
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


def run_ekf_baseline(
    *,
    all_meas,
    stations,
    Xtrue_meas,
    x0,
    P0,
    R,
    Q_accel,
    mu,
    J2,
    J3,
):
    ekf = ExtendedKalmanFilter(
        x0=x0,
        P0=P0,
        R=R,
        Q=Q_accel,
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


def make_postfit_difference_plot(t_s: np.ndarray, dpf: np.ndarray, outdir: Path, filename: str, title: str) -> None:
    t_hr = as_hours(t_s)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(t_hr, dpf[:, 0], ".", markersize=2)
    ax[0].axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
    ax[0].set_ylabel("d rho [m]")
    ax[0].set_title(title)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t_hr, dpf[:, 1], ".", markersize=2)
    ax[1].axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
    ax[1].set_ylabel("d rhodot [m/s]")
    ax[1].set_xlabel("Time [hours]")
    ax[1].grid(True, alpha=0.3)

    savefig(fig, outdir / filename)


def make_state_error_difference_plot(t_s: np.ndarray, de: np.ndarray, outdir: Path) -> None:
    t_hr = as_hours(t_s)
    labels = ["dx [m]", "dy [m]", "dz [m]", "dvx [m/s]", "dvy [m/s]", "dvz [m/s]"]

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axs = axs.flatten()
    for i in range(6):
        axs[i].plot(t_hr, de[:, i], ".", markersize=2)
        axs[i].axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
    axs[-2].set_xlabel("Time [hours]")
    axs[-1].set_xlabel("Time [hours]")
    fig.suptitle("State-Error Difference: UKF - EKF (J3 OFF)")

    savefig(fig, outdir / "state_error_difference_ukf_minus_ekf.png")


def make_state_error_difference_plot_zoom_no_first_hour(t_s: np.ndarray, de: np.ndarray, outdir: Path) -> None:
    mask = np.asarray(t_s, dtype=float) >= 3600.0
    if not np.any(mask):
        return

    t_hr = as_hours(np.asarray(t_s, dtype=float)[mask])
    de_z = np.asarray(de, dtype=float)[mask, :]
    labels = ["dx [m]", "dy [m]", "dz [m]", "dvx [m/s]", "dvy [m/s]", "dvz [m/s]"]

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axs = axs.flatten()
    for i in range(6):
        axs[i].plot(t_hr, de_z[:, i], ".", markersize=2)
        axs[i].axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
    axs[-2].set_xlabel("Time [hours]")
    axs[-1].set_xlabel("Time [hours]")
    fig.suptitle("State-Error Difference: UKF - EKF (J3 OFF), t >= 1 hr")

    savefig(fig, outdir / "state_error_difference_ukf_minus_ekf_zoom_t_ge_1h.png")


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0 = load_hw3_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    sigma_q_m_s2 = get_optimal_sigma_q_m_s2()
    Q_accel = build_q_accel_cov_km(sigma_q_m_s2)

    out_ukf = run_ukf_baseline(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0=x0,
        P0=P0,
        R=R,
        Q_accel=Q_accel,
        mu=mu,
        J2=J2,
        J3=J3,
    )
    out_ekf = run_ekf_baseline(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0=x0,
        P0=P0,
        R=R,
        Q_accel=Q_accel,
        mu=mu,
        J2=J2,
        J3=J3,
    )

    res_ukf = convert_filter_output_km_to_m_for_plotting(out_ukf, R_km=R)
    res_ekf = convert_filter_output_km_to_m_for_plotting(out_ekf, R_km=R)

    t_ukf = np.asarray(res_ukf.t_meas, dtype=float)
    t_ekf = np.asarray(res_ekf.t_meas, dtype=float)
    if t_ukf.shape != t_ekf.shape or not np.allclose(t_ukf, t_ekf):
        raise RuntimeError("UKF and EKF timelines are not aligned; cannot compute pointwise differences.")

    dpost_lin = (
        np.asarray(res_ukf.postfit_resids_linear_final, dtype=float)
        - np.asarray(res_ekf.postfit_resids_linear_final, dtype=float)
    )
    dstate = (
        np.asarray(res_ukf.state_error_meas, dtype=float)
        - np.asarray(res_ekf.state_error_meas, dtype=float)
    )

    plot_dir = Path(__file__).resolve().parent / "Plots" / "Comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    make_postfit_difference_plot(
        t_s=t_ukf,
        dpf=dpost_lin,
        outdir=plot_dir,
        filename="postfit_difference_linear_ukf_minus_ekf.png",
        title="Linear Postfit Difference: UKF - EKF (J3 OFF)",
    )
    make_state_error_difference_plot(t_s=t_ukf, de=dstate, outdir=plot_dir)
    make_state_error_difference_plot_zoom_no_first_hour(t_s=t_ukf, de=dstate, outdir=plot_dir)

    print(f"Using optimal SNC sigma from HW3 EKF sweep: {sigma_q_m_s2:.6e} m/s^2")
    print(f"Saved UKF-EKF comparison plots to: {plot_dir}")


if __name__ == "__main__":
    main()
