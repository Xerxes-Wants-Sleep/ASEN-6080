from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .common import as_hours, savefig
from .post_processing import BatchPostProcessResult


def make_state_errors_rsw_plot(result: BatchPostProcessResult, outdir: Path, show: bool = False) -> None:
    if result.pos_err_rsw is None or result.Pdiag_pos_rsw is None:
        raise ValueError("RSW quantities not available. Provide truth so RSW can be computed.")

    t_hr = as_hours(result.t_meas)
    e_rsw = result.pos_err_rsw
    sig3_rsw = 3.0 * np.sqrt(np.maximum(result.Pdiag_pos_rsw, 0.0))

    labels = ["R", "S", "W"]
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i in range(3):
        ax[i].plot(t_hr, e_rsw[:, i], ".", markersize=2)
        ax[i].plot(t_hr, sig3_rsw[:, i], "r")
        ax[i].plot(t_hr, -sig3_rsw[:, i], "r")
        ax[i].set_ylabel(f"{labels[i]} [km]")
    ax[0].set_title("RSW Position Errors with ±3σ")
    ax[-1].set_xlabel("Time [hours]")
    savefig(fig, outdir / "state_errors_pos_rsw.png", show=show)
