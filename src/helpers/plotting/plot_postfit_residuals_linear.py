from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

from .common import station_masks, savefig
from .post_processing import BatchPostProcessResult


def _qqplot(ax, data, label=None):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return
    mu = float(np.mean(data))
    sigma = float(np.std(data))
    if sigma == 0.0:
        return
    z = (data - mu) / sigma
    q = (np.arange(1, z.size + 1) - 0.5) / z.size
    theo = norm.ppf(q)
    ax.plot(theo, np.sort(z), ".", label=label)
    lim = np.array(ax.get_xlim() + ax.get_ylim())
    lim_min = np.min(lim)
    lim_max = np.max(lim)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)


def _residuals_1x3_row(fig, gs_row, t_sec, res, masks, title, ylab):
    ax_ts = fig.add_subplot(gs_row[0])
    ax_hist = fig.add_subplot(gs_row[1])
    ax_qq = fig.add_subplot(gs_row[2])

    for s, m in masks.items():
        ax_ts.plot(t_sec[m], res[m], ".", label=s, markersize=2)
    ax_ts.axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
    ax_ts.set_title(title)
    ax_ts.set_ylabel(ylab)
    ax_ts.set_xlabel("Time (s)")
    ax_ts.legend()

    r = res[np.isfinite(res)]
    mu = float(np.mean(r)) if r.size else 0.0
    sigma = float(np.std(r)) if r.size else 0.0
    ax_hist.hist(r, bins=30, orientation="horizontal", color="C0", alpha=0.7, edgecolor="none")
    ax_hist.set_title(rf"$\mu$={mu:.2e}, $\sigma$={sigma:.2e}")
    ax_hist.set_xlabel("Count")
    ax_hist.set_yticklabels([])

    _qqplot(ax_qq, r)
    ax_qq.set_title("Q-Q Plot")
    ax_qq.set_xlabel("Theoretical Quantiles")
    ax_qq.set_ylabel("Sample Quantiles")


def make_postfit_residuals_linear_plot(result: BatchPostProcessResult, outdir: Path, show: bool = False) -> None:
    pf = result.postfit_resids_linear_final
    if pf is None:
        raise ValueError("postfit_resids_linear_final is None. Pass info=... into run_batch_post_processing().")

    t_sec = np.asarray(result.t_meas, dtype=float)
    masks = station_masks(result.station_meas)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.8, 1.1, 1.1], hspace=0.35, wspace=0.3)

    _residuals_1x3_row(
        fig,
        [gs[0, 0], gs[0, 1], gs[0, 2]],
        t_sec,
        pf[:, 0],
        masks,
        "Range",
        "Range Residual (m)",
    )
    _residuals_1x3_row(
        fig,
        [gs[1, 0], gs[1, 1], gs[1, 2]],
        t_sec,
        pf[:, 1],
        masks,
        "Range-rate",
        "Range-rate Residual (m/s)",
    )

    savefig(fig, outdir / "postfit_residuals_linear.png", show=show, tight_layout=False)
