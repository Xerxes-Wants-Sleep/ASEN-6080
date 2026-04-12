from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _ellipse_points_2d(P2: np.ndarray, mu2: np.ndarray, n_sigma: float, n_pts: int = 200) -> np.ndarray:
    evals, evecs = np.linalg.eigh(np.asarray(P2, dtype=float))
    evals = np.clip(evals, 0.0, None)
    radii = float(n_sigma) * np.sqrt(evals)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    return (evecs @ (radii[:, None] * circle)) + np.asarray(mu2, dtype=float).reshape(2, 1)


def corner_plot_mc(
    *,
    prop_traj: np.ndarray,
    monte_traj: np.ndarray,
    P: np.ndarray | None = None,
    title_text: str = "",
    labels: list[str] | None = None,
    border_frac: float = 0.08,
) -> plt.Figure:
    """
    Corner plot for 6-state Monte Carlo samples.

    Parameters
    ----------
    prop_traj : (6,) array_like
        Nominal/propagated state at one epoch.
    monte_traj : (6,N) array_like
        MC states at one epoch (variables in rows, cases in columns).
    P : (6,6) array_like, optional
        Covariance for monte_traj. If None, computed from samples.
    title_text : str
        Figure title.
    labels : list[str], optional
        Axis labels for the 6 states.
    """
    X = np.asarray(monte_traj, dtype=float)
    if X.shape[0] != 6:
        raise ValueError(f"monte_traj must have shape (6,N). Got {X.shape}.")

    x_nom = np.asarray(prop_traj, dtype=float).reshape(6)
    mu = np.mean(X, axis=1)
    cov = np.asarray(P, dtype=float) if P is not None else np.cov(X)
    sig = np.sqrt(np.maximum(np.diag(cov), 0.0))

    if labels is None:
        labels = ["X", "Y", "Z", "Xdot", "Ydot", "Zdot"]

    fig, axes = plt.subplots(6, 6, figsize=(16, 16))
    fig.suptitle(title_text)

    # Global per-state limits used for MATLAB-like linked axes.
    # Start from MC spread + mean + nominal, then expand using ellipse extents.
    v_lo = np.minimum(np.min(X, axis=1), np.minimum(mu, x_nom))
    v_hi = np.maximum(np.max(X, axis=1), np.maximum(mu, x_nom))

    for i in range(6):
        for j in range(6):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            ax.grid(True, alpha=0.25)

            if i == j:
                xj = X[j, :]
                ax.hist(xj, bins=40, density=True, color="b", alpha=0.45)
                sigma_j = sig[j]
                span = 3.0 * sigma_j if sigma_j > 0.0 else 1.0
                xs = np.linspace(mu[j] - span, mu[j] + span, 600)
                sigma_safe = max(sigma_j, 1e-14)
                pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_safe)) * np.exp(
                    -0.5 * ((xs - mu[j]) / sigma_safe) ** 2
                )
                ax.plot(xs, pdf, "m--", linewidth=2.0)
                ax.axvline(x_nom[j], color="k", linestyle="--", linewidth=2.0)
                ax.set_xlabel(labels[j])
                v_lo[j] = min(v_lo[j], float(xs[0]))
                v_hi[j] = max(v_hi[j], float(xs[-1]))
            else:
                idx = [j, i]
                P2 = cov[np.ix_(idx, idx)]
                mu2 = mu[idx]

                e1 = _ellipse_points_2d(P2, mu2, 1.0)
                e2 = _ellipse_points_2d(P2, mu2, 2.0)
                e3 = _ellipse_points_2d(P2, mu2, 3.0)

                ax.plot(e1[0, :], e1[1, :], "b--", linewidth=1.2)
                ax.plot(e2[0, :], e2[1, :], "r--", linewidth=1.2)
                ax.plot(e3[0, :], e3[1, :], "g--", linewidth=1.2)
                ax.scatter(X[j, :], X[i, :], s=3, c="k", alpha=0.25)
                ax.scatter(mu[j], mu[i], s=30, marker="s", c="k")
                ax.scatter(x_nom[j], x_nom[i], s=30, marker="^", c="k")
                ax.set_xlabel(labels[j])
                ax.set_ylabel(labels[i])
                # Ensure linked limits include full 3-sigma ellipse.
                v_lo[j] = min(v_lo[j], float(np.min(e3[0, :])))
                v_hi[j] = max(v_hi[j], float(np.max(e3[0, :])))
                v_lo[i] = min(v_lo[i], float(np.min(e3[1, :])))
                v_hi[i] = max(v_hi[i], float(np.max(e3[1, :])))

    # Link scales per state across panels (MATLAB-like linkaxes behavior).
    for k in range(6):
        lo = float(v_lo[k])
        hi = float(v_hi[k])
        pad = float(border_frac) * max(hi - lo, 1e-12)
        lo -= pad
        hi += pad
        for i in range(k, 6):
            axes[i, k].set_xlim(lo, hi)

    for k in range(1, 6):
        lo = float(v_lo[k])
        hi = float(v_hi[k])
        pad = float(border_frac) * max(hi - lo, 1e-12)
        lo -= pad
        hi += pad
        for j in range(k):
            axes[k, j].set_ylim(lo, hi)

    legend_handles = [
        Line2D([0], [0], color="m", linestyle="--", linewidth=2.0, label="State Component PDF"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=2.0, label="State Component Nominal Value"),
        Line2D([0], [0], color="b", linestyle="--", linewidth=1.2, label="1sigma Ellipse"),
        Line2D([0], [0], color="r", linestyle="--", linewidth=1.2, label="2sigma Ellipse"),
        Line2D([0], [0], color="g", linestyle="--", linewidth=1.2, label="3sigma Ellipse"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=5,
            alpha=0.5,
            label="MC Points",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=6,
            label="MC Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=6,
            label="Propagated Trajectory Point",
        ),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.97))
    fig.subplots_adjust(top=0.92, wspace=0.35, hspace=0.35)
    return fig
