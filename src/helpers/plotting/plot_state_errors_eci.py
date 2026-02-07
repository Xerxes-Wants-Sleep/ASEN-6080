from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .common import as_hours, savefig
from .post_processing import BatchPostProcessResult


def make_state_errors_eci_plots(
    result: BatchPostProcessResult,
    outdir: Path,
    show: bool = False,
    zoom_t0_s: float = 30000.0,
) -> None:
    if result.state_error_meas is None:
        raise ValueError("No truth provided; state_error_meas is None.")
    if result.P_meas is None:
        raise ValueError("No covariance provided; P_meas is None.")

    t_s = np.asarray(result.t_meas, dtype=float)
    t_hr = as_hours(t_s)
    e = result.state_error_meas  # (m,6)

    # ±3σ from covariance
    sig3 = 3.0 * np.sqrt(np.maximum(np.diagonal(result.P_meas, axis1=1, axis2=2), 0.0))  # (m,6)

    # -----------------------------
    # FULL plots (keep exactly as before)
    # -----------------------------
    labels = ["x", "y", "z"]
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i in range(3):
        ax[i].plot(t_hr, e[:, i], ".", markersize=2)
        ax[i].plot(t_hr, sig3[:, i], "r")
        ax[i].plot(t_hr, -sig3[:, i], "r")
        ax[i].set_ylabel(f"{labels[i]} [km]")
    ax[0].set_title("ECI Position Errors with ±3σ")
    ax[-1].set_xlabel("Time [hours]")
    savefig(fig, outdir / "state_errors_pos_eci.png", show=show)

    labels_v = ["vx", "vy", "vz"]
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i in range(3):
        k = i + 3
        ax[i].plot(t_hr, e[:, k], ".", markersize=2)
        ax[i].plot(t_hr, sig3[:, k], "r")
        ax[i].plot(t_hr, -sig3[:, k], "r")
        ax[i].set_ylabel(f"{labels_v[i]} [km/s]")
    ax[0].set_title("ECI Velocity Errors with ±3σ")
    ax[-1].set_xlabel("Time [hours]")
    savefig(fig, outdir / "state_errors_vel_eci.png", show=show)

    # -----------------------------
    # ZOOM plots: t >= zoom_t0_s
    # -----------------------------
    mask = t_s >= float(zoom_t0_s)
    if not np.any(mask):
        # Nothing to plot in zoom window; silently skip
        return

    t_hr_z = as_hours(t_s[mask])
    e_z = e[mask, :]
    sig3_z = sig3[mask, :]

    # Position zoom
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i in range(3):
        ax[i].plot(t_hr_z, e_z[:, i], ".", markersize=2)
        ax[i].plot(t_hr_z, sig3_z[:, i], "r")
        ax[i].plot(t_hr_z, -sig3_z[:, i], "r")
        ax[i].set_ylabel(f"{labels[i]} [km]")
    ax[0].set_title(f"ECI Position Errors with ±3σ (t ≥ {int(zoom_t0_s)} s)")
    ax[-1].set_xlabel("Time [hours]")
    savefig(fig, outdir / f"state_errors_pos_eci_zoom_t_ge_{int(zoom_t0_s)}.png", show=show)

    # Velocity zoom
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i in range(3):
        k = i + 3
        ax[i].plot(t_hr_z, e_z[:, k], ".", markersize=2)
        ax[i].plot(t_hr_z, sig3_z[:, k], "r")
        ax[i].plot(t_hr_z, -sig3_z[:, k], "r")
        ax[i].set_ylabel(f"{labels_v[i]} [km/s]")
    ax[0].set_title(f"ECI Velocity Errors with ±3σ (t ≥ {int(zoom_t0_s)} s)")
    ax[-1].set_xlabel("Time [hours]")
    savefig(fig, outdir / f"state_errors_vel_eci_zoom_t_ge_{int(zoom_t0_s)}.png", show=show)
