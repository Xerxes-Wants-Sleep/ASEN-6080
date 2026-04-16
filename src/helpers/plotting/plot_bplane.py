from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .common import savefig


def _cov_rt_from_inputs(
    *,
    P_bplane: np.ndarray | None,
    sigma_r_km: float | None,
    sigma_t_km: float | None,
    sigma_rt_km2: float | None,
) -> np.ndarray | None:
    if P_bplane is not None:
        P = np.asarray(P_bplane, dtype=float)
        if P.shape == (7, 7):
            return np.array(
                [
                    [P[2, 2], P[1, 2]],  # [R,R], [R,T]
                    [P[1, 2], P[1, 1]],  # [T,R], [T,T]
                ],
                dtype=float,
            )
        if P.shape == (2, 2):
            return P
        raise ValueError("P_bplane must be shape (7,7) from calc_bplane or shape (2,2) in [R,T].")

    if sigma_r_km is None or sigma_t_km is None:
        return None
    cov_rt = 0.0 if sigma_rt_km2 is None else float(sigma_rt_km2)
    return np.array(
        [
            [float(sigma_r_km) ** 2, cov_rt],
            [cov_rt, float(sigma_t_km) ** 2],
        ],
        dtype=float,
    )


def _ellipse_xy(cov_2x2: np.ndarray, n_sigma: float, n_pts: int = 361) -> tuple[np.ndarray, np.ndarray]:
    cov = np.asarray(cov_2x2, dtype=float).reshape(2, 2)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    radii = n_sigma * np.sqrt(vals)

    th = np.linspace(0.0, 2.0 * np.pi, n_pts)
    circle = np.vstack((np.cos(th), np.sin(th)))
    ellipse = vecs @ np.diag(radii) @ circle
    return ellipse[0, :], ellipse[1, :]


def make_bplane_plot(
    *,
    BdotR_km: float,
    BdotT_km: float,
    outdir: Path,
    filename: str = "bplane.png",
    title: str = "B-Plane",
    P_bplane: np.ndarray | None = None,
    sigma_r_km: float | None = None,
    sigma_t_km: float | None = None,
    sigma_rt_km2: float | None = None,
    n_sigma_levels: Iterable[float] = (1.0, 2.0, 3.0),
    samples_rt_km: np.ndarray | None = None,
    show_origin_axes: bool = False,
    show: bool = False,
) -> dict[str, float]:
    """
    Plot a B-plane point and optional covariance ellipses.

    Axes are [BdotR, BdotT] in km.
    Covariance can be passed as:
    - `P_bplane` shape (7,7) from `calc_bplane` (uses indices R=2, T=1), or
    - `P_bplane` shape (2,2) in [R,T] order, or
    - sigma terms (`sigma_r_km`, `sigma_t_km`, optional `sigma_rt_km2`).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    x_all = [float(BdotR_km)]
    y_all = [float(BdotT_km)]

    ax.plot(BdotR_km, BdotT_km, "o", markersize=6, color="k", label="Nominal B-Plane Point")

    if samples_rt_km is not None:
        samples = np.asarray(samples_rt_km, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("samples_rt_km must have shape (N,2) in [BdotR, BdotT].")
        ax.plot(samples[:, 0], samples[:, 1], ".", markersize=2, alpha=0.35, label="Samples")
        x_all.extend(samples[:, 0].tolist())
        y_all.extend(samples[:, 1].tolist())

    cov_rt = _cov_rt_from_inputs(
        P_bplane=P_bplane,
        sigma_r_km=sigma_r_km,
        sigma_t_km=sigma_t_km,
        sigma_rt_km2=sigma_rt_km2,
    )
    if cov_rt is not None:
        for k in n_sigma_levels:
            ex, ey = _ellipse_xy(cov_rt, n_sigma=float(k))
            xk = BdotR_km + ex
            yk = BdotT_km + ey
            ax.plot(xk, yk, linewidth=1.2, label=f"{float(k):g} Sigma")
            x_all.extend(xk.tolist())
            y_all.extend(yk.tolist())

    x_min = min(x_all)
    x_max = max(x_all)
    y_min = min(y_all)
    y_max = max(y_all)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    pad_x = 0.08 * x_span
    pad_y = 0.08 * y_span
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    if show_origin_axes:
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        if y_lo <= 0.0 <= y_hi:
            ax.axhline(0.0, color="0.75", linewidth=0.8)
        if x_lo <= 0.0 <= x_hi:
            ax.axvline(0.0, color="0.75", linewidth=0.8)

    ax.set_xlabel("B Dot R [km]")
    ax.set_ylabel("B Dot T [km]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="best")

    savefig(fig, outdir / filename, show=show)

    out = {"BdotR_km": float(BdotR_km), "BdotT_km": float(BdotT_km)}
    if cov_rt is not None:
        sig_r = float(np.sqrt(max(cov_rt[0, 0], 0.0)))
        sig_t = float(np.sqrt(max(cov_rt[1, 1], 0.0)))
        denom = sig_r * sig_t
        corr = float(cov_rt[0, 1] / denom) if denom > 0.0 else float("nan")
        out.update({"sigma_R_km": sig_r, "sigma_T_km": sig_t, "corr_RT": corr})
    return out


def make_bplane_plot_from_calc_result(
    calc_result: tuple,
    *,
    outdir: Path,
    filename: str = "bplane.png",
    title: str = "B-Plane",
    n_sigma_levels: Iterable[float] = (1.0, 2.0, 3.0),
    samples_rt_km: np.ndarray | None = None,
    show_origin_axes: bool = False,
    show: bool = False,
) -> dict[str, float]:
    """
    Convenience wrapper for `src.Functions.calcB_plane.calc_bplane` output tuple.
    """
    BdotR_km = float(calc_result[0])
    BdotT_km = float(calc_result[1])
    P_bplane = np.asarray(calc_result[6], dtype=float)
    return make_bplane_plot(
        BdotR_km=BdotR_km,
        BdotT_km=BdotT_km,
        P_bplane=P_bplane,
        outdir=outdir,
        filename=filename,
        title=title,
        n_sigma_levels=n_sigma_levels,
        samples_rt_km=samples_rt_km,
        show_origin_axes=show_origin_axes,
        show=show,
    )
