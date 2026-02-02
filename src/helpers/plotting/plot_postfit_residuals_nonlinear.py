from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path

from .common import as_hours, station_masks, savefig
from .post_processing import BatchPostProcessResult


def make_postfit_residuals_nonlinear_plot(result: BatchPostProcessResult, outdir: Path, show: bool = False) -> None:
    pf = result.postfit_resids_meas

    t_hr = as_hours(result.t_meas)
    masks = station_masks(result.station_meas)

    fig, ax = plt.subplots(2, 1, sharex=True)
    for s, m in masks.items():
        ax[0].plot(t_hr[m], pf[m, 0], ".", label=s, markersize=2)
        ax[1].plot(t_hr[m], pf[m, 1], ".", label=s, markersize=2)

    ax[0].set_title("Nonlinear Post-fit Residuals (Y - h(x̂(t)))")
    ax[0].set_ylabel("Range [km]")
    ax[1].set_ylabel("Range-rate [km/s]")
    ax[1].set_xlabel("Time [hours]")
    ax[0].legend()

    savefig(fig, outdir / "postfit_residuals_nonlinear.png", show=show)
