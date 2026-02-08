from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .common import as_hours, savefig
from .post_processing import BatchPostProcessResult


def make_trace_cov_pos_vel_plot(
    result: BatchPostProcessResult,
    outdir: Path,
    show: bool = False,
    filename: str = "trace_cov_pos_vel.png",
    title: str | None = None,
    length_unit: str = "m",
) -> None:
    t_hr = as_hours(result.t_meas)

    P = np.asarray(result.P_meas, dtype=float)
    if P.ndim != 3 or P.shape[1] < 6 or P.shape[2] < 6:
        raise ValueError("P_meas must have shape (m, n, n) with n >= 6.")

    unit = str(length_unit).strip().lower()
    if unit == "m":
        scale = 1.0
        ylabel = "trace(P) [m^2, (m/s)^2]"
    elif unit == "km":
        scale = 1.0e-3
        ylabel = "trace(P) [km^2, (km/s)^2]"
    else:
        raise ValueError("length_unit must be 'km' or 'm'")

    P = P * (scale ** 2)

    # trace of position and velocity blocks
    tr_pos = np.sum(np.maximum(np.diagonal(P[:, 0:3, 0:3], axis1=1, axis2=2), 0.0), axis=1)
    tr_vel = np.sum(np.maximum(np.diagonal(P[:, 3:6, 3:6], axis1=1, axis2=2), 0.0), axis=1)

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10, 4))

    ax[0].semilogy(t_hr, tr_pos, ".", markersize=2)
    ax[0].set_xlabel("Time [hours]")
    ax[0].set_ylabel(ylabel)
    ax[0].set_title("trace(P_pos) (log scale)")

    ax[1].semilogy(t_hr, tr_vel, ".", markersize=2)
    ax[1].set_xlabel("Time [hours]")
    ax[1].set_ylabel(ylabel)
    ax[1].set_title("trace(P_vel) (log scale)")

    if title is None:
        fig.suptitle("Trace of Covariance (pos/vel)")
    else:
        fig.suptitle(title)

    savefig(fig, outdir / filename, show=show)
