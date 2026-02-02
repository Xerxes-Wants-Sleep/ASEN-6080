from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .common import as_hours, station_masks, savefig
from .post_processing import BatchPostProcessResult


def make_prefit_residuals_plot(result: BatchPostProcessResult, outdir: Path, show: bool = False) -> None:
    pre = result.prefit_resids_final
    if pre is None:
        raise ValueError("prefit_resids_final is None. Pass info=... into run_batch_post_processing().")

    t_hr = as_hours(result.t_meas)
    masks = station_masks(result.station_meas)

    fig, ax = plt.subplots(2, 1, sharex=True)
    for s, m in masks.items():
        ax[0].plot(t_hr[m], pre[m, 0], ".", label=s, markersize=2)
        ax[1].plot(t_hr[m], pre[m, 1], ".", label=s, markersize=2)

    ax[0].set_title("Pre-fit Residuals (final iteration)")
    ax[0].set_ylabel("Range [km]")
    ax[1].set_ylabel("Range-rate [km/s]")
    ax[1].set_xlabel("Time [hours]")
    ax[0].legend()

    savefig(fig, outdir / "prefit_residuals.png", show=show)
