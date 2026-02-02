from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .common import as_hours, savefig
from .post_processing import BatchPostProcessResult


def make_state_errors_eci_plots(result: BatchPostProcessResult, outdir: Path, show: bool = False) -> None:
    if result.state_error_meas is None:
        raise ValueError("No truth provided; state_error_meas is None.")

    t_hr = as_hours(result.t_meas)
    e = result.state_error_meas  # (m,6)

    sig3 = 3.0 * np.sqrt(np.maximum(np.diagonal(result.P_meas, axis1=1, axis2=2), 0.0))  # (m,6)

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
