from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .common import as_hours, savefig
from .post_processing import BatchPostProcessResult


def make_cov_diag_log_plot(result: BatchPostProcessResult, outdir: Path, show: bool = False) -> None:
    t_hr = as_hours(result.t_meas)
    Pdiag = np.maximum(np.diagonal(result.P_meas, axis1=1, axis2=2), 0.0)
    sig = np.sqrt(Pdiag)

    fig = plt.figure()
    for i in range(6):
        plt.semilogy(t_hr, sig[:, i], "-o", markersize=1)
    plt.xlabel("Time [hours]")
    plt.ylabel(r"$\sigma$ [km, km/s]")
    plt.title("Covariance Diagonal (1σ) in Log Scale")
    plt.legend(["x", "y", "z", "vx", "vy", "vz"])

    savefig(fig, outdir / "cov_diag_log.png", show=show)
