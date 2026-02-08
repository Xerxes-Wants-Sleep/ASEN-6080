from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .common import as_hours, savefig
from .post_processing import BatchPostProcessResult


def make_trace_cov_plot(result: BatchPostProcessResult, outdir: Path, show: bool = False) -> None:
    t_hr = as_hours(result.t_meas)
    Pdiag = np.maximum(np.diagonal(result.P_meas, axis1=1, axis2=2), 0.0)
    tr = np.sum(Pdiag, axis=1)

    fig = plt.figure()
    plt.semilogy(t_hr, tr, ".", markersize=2)
    plt.xlabel("Time [hours]")
    plt.ylabel("trace(P) [m^2, (m/s)^2]")
    plt.title("Trace of Covariance (log scale)")

    savefig(fig, outdir / "trace_cov.png", show=show)
