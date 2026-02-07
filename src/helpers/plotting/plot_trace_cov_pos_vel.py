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
) -> None:
    t_hr = as_hours(result.t_meas)

    P = np.asarray(result.P_meas, dtype=float)
    if P.ndim != 3 or P.shape[1] < 6 or P.shape[2] < 6:
        raise ValueError("P_meas must have shape (m, n, n) with n >= 6.")

    # trace of position and velocity blocks
    tr_pos = np.sum(np.maximum(np.diagonal(P[:, 0:3, 0:3], axis1=1, axis2=2), 0.0), axis=1)
    tr_vel = np.sum(np.maximum(np.diagonal(P[:, 3:6, 3:6], axis1=1, axis2=2), 0.0), axis=1)

    fig = plt.figure()
    plt.semilogy(t_hr, tr_pos, "-o", markersize=2, label="pos trace")
    plt.semilogy(t_hr, tr_vel, "-o", markersize=2, label="vel trace")
    plt.xlabel("Time [hours]")
    plt.ylabel("trace(P)")
    if title is None:
        plt.title("Trace of Covariance (pos/vel) (log scale)")
    else:
        plt.title(title)
    plt.legend()

    savefig(fig, outdir / filename, show=show)
