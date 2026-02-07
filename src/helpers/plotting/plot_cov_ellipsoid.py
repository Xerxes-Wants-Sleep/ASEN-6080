from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

from .common import savefig
from .post_processing import BatchPostProcessResult


def _get_ellipsoid_points(
    P_sub: np.ndarray,
    sigma_val: float,
    num_points: int = 800,
    *,
    scatter_sigma: float = 1.0,
    seed: int | None = None,
):
    """
    Returns scattered XYZ points on the ellipsoid surface and principal axes.
    """
    eig_vals, eig_vecs = np.linalg.eigh(P_sub)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    std_devs = np.sqrt(np.abs(eig_vals))      # 1-sigma along principal axes
    radii = sigma_val * std_devs              # radii for the sigma shell you’re plotting

    # --- Interior Gaussian cloud (matches friend's A1 @ z) ---
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((3, num_points))  # N(0, I), NOT normalized
    ellipsoid_points = (eig_vecs @ (std_devs[:, np.newaxis] * z)) * scatter_sigma

    return ellipsoid_points, eig_vecs, radii, std_devs


def _get_ellipsoid_surface(P_sub: np.ndarray, sigma_val: float):
    eig_vals, eig_vecs = np.linalg.eigh(P_sub)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    radii = sigma_val * np.sqrt(np.abs(eig_vals))

    u = np.linspace(0.0, 2.0 * np.pi, 25)
    v = np.linspace(0.0, np.pi, 25)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    ellipsoid_points = np.zeros((len(u), len(v), 3), dtype=float)
    for i in range(len(u)):
        for j in range(len(v)):
            point = np.array([x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]])
            ellipsoid_points[i, j, :] = eig_vecs @ (point * radii)
    return ellipsoid_points


def plot_cov_ellipsoid(
    result: BatchPostProcessResult,
    outdir: Path,
    *,
    show: bool = False,
    filename: str = "final_covariance_ellipsoids_stats.png",
    title_prefix: str | None = None,
) -> None:
    """
    Plots nested 1, 2, and 3-sigma 3D Position AND Velocity covariance ellipsoids.
    """
    P_hist = np.asarray(result.P_meas, dtype=float)
    if P_hist.size == 0:
        return

    P_final = P_hist[-1]

    # Convert from km and km/s to meters and m/s for nicer axes
    scale_pos = 1.0e3
    scale_vel = 1.0e3

    P_pos = P_final[0:3, 0:3] * (scale_pos ** 2)
    P_vel = P_final[3:6, 3:6] * (scale_vel ** 2)

    pos_unit = "m"
    vel_unit = "m/s"

    fig = plt.figure(figsize=(16, 7))
    sigmas = [1, 2, 3]
    alphas = {1: 0.3, 2: 0.15, 3: 0.05}

    # --- Position subplot ---
    ax1 = fig.add_subplot(121, projection="3d")
    max_pos_r = 0.0
    pos_std_devs = []

    pos_vec_handles = []
    pos_vec_colors = ["b", "r", "k"]

    for s in sigmas:
        surf = _get_ellipsoid_surface(P_pos, s)
        ax1.plot_surface(
            surf[:, :, 0],
            surf[:, :, 1],
            surf[:, :, 2],
            rstride=1,
            cstride=1,
            color="blue",
            alpha=alphas[s],
            linewidth=0,
            shade=True,
        )

        if s == 3:
            pts, vecs, radii, std_devs = _get_ellipsoid_points(
                P_pos, s, num_points=800, scatter_sigma=1.0, seed=0
)
            max_pos_r = float(np.max(radii))
            pos_std_devs = std_devs

            ax1.scatter(
                pts[0, :],
                pts[1, :],
                pts[2, :],
                c="grey",
                s=2,
                alpha=0.6,
                depthshade=False,
            )

            for i in range(3):
                v = vecs[:, i] * radii[i]
                line = ax1.plot(
                    [-v[0], v[0]],
                    [-v[1], v[1]],
                    [-v[2], v[2]],
                    color=pos_vec_colors[i],
                    lw=2,
                    zorder=10,
                )[0]
                pos_vec_handles.append(line)

    ax1.set_xlabel(f"X ({pos_unit})")
    ax1.set_ylabel(f"Y ({pos_unit})")
    ax1.set_zlabel(f"Z ({pos_unit})")

    if len(pos_std_devs) == 0:
        pos_std_devs = [0.0, 0.0, 0.0]

    title_str_pos = (
        f"Position Covariance (1, 2, 3$\\sigma$)\n"
        f"$\\sigma_{{max}}$: {pos_std_devs[0]:.4f} {pos_unit} | "
        f"$\\sigma_{{min}}$: {pos_std_devs[2]:.4f} {pos_unit}"
    )
    if title_prefix:
        title_str_pos = f"{title_prefix}\n{title_str_pos}"
    ax1.set_title(title_str_pos)

    limit_p = max_pos_r * 1.2 if max_pos_r > 0.0 else 1.0
    ax1.set_xlim(-limit_p, limit_p)
    ax1.set_ylim(-limit_p, limit_p)
    ax1.set_zlim(-limit_p, limit_p)

    # Legend: sigma rings (position)
    pos_handles = [
        mpatches.Patch(color="blue", alpha=alphas[1], label="1σ"),
        mpatches.Patch(color="blue", alpha=alphas[2], label="2σ"),
        mpatches.Patch(color="blue", alpha=alphas[3], label="3σ"),
    ]
    if len(pos_vec_handles) >= 3:
        pos_handles += [
            pos_vec_handles[0],
            pos_vec_handles[1],
            pos_vec_handles[2],
        ]
        pos_handles[-3].set_label("v1")
        pos_handles[-2].set_label("v2")
        pos_handles[-1].set_label("v3")
    ax1.legend(handles=pos_handles, loc="upper right")

    # --- Velocity subplot ---
    ax2 = fig.add_subplot(122, projection="3d")
    max_vel_r = 0.0
    vel_std_devs = []

    vel_vec_handles = []
    vel_vec_colors = ["b", "r", "k"]

    for s in sigmas:
        surf = _get_ellipsoid_surface(P_vel, s)
        ax2.plot_surface(
            surf[:, :, 0],
            surf[:, :, 1],
            surf[:, :, 2],
            rstride=1,
            cstride=1,
            color="green",
            alpha=alphas[s],
            linewidth=0,
            shade=True,
        )

        if s == 3:
            pts, vecs, radii, std_devs = _get_ellipsoid_points(
                P_vel, s, num_points=800, scatter_sigma=1.0, seed=0
)
            max_vel_r = float(np.max(radii))
            vel_std_devs = std_devs

            ax2.scatter(
                pts[0, :],
                pts[1, :],
                pts[2, :],
                c="grey",
                s=2,
                alpha=0.6,
                depthshade=False,
            )

            for i in range(3):
                v = vecs[:, i] * radii[i]
                line = ax2.plot(
                    [-v[0], v[0]],
                    [-v[1], v[1]],
                    [-v[2], v[2]],
                    color=vel_vec_colors[i],
                    lw=2,
                    zorder=10,
                )[0]
                vel_vec_handles.append(line)

    ax2.set_xlabel(f"Vx ({vel_unit})")
    ax2.set_ylabel(f"Vy ({vel_unit})")
    ax2.set_zlabel(f"Vz ({vel_unit})")

    if len(vel_std_devs) == 0:
        vel_std_devs = [0.0, 0.0, 0.0]

    title_str_vel = (
        f"Velocity Covariance (1, 2, 3$\\sigma$)\n"
        f"$\\sigma_{{max}}$: {vel_std_devs[0]:.4e} {vel_unit} | "
        f"$\\sigma_{{min}}$: {vel_std_devs[2]:.4e} {vel_unit}"
    )
    if title_prefix:
        title_str_vel = f"{title_prefix}\n{title_str_vel}"
    ax2.set_title(title_str_vel)

    limit_v = max_vel_r * 1.2 if max_vel_r > 0.0 else 1.0
    ax2.set_xlim(-limit_v, limit_v)
    ax2.set_ylim(-limit_v, limit_v)
    ax2.set_zlim(-limit_v, limit_v)

    vel_handles = [
        mpatches.Patch(color="green", alpha=alphas[1], label="1σ"),
        mpatches.Patch(color="green", alpha=alphas[2], label="2σ"),
        mpatches.Patch(color="green", alpha=alphas[3], label="3σ"),
    ]
    if len(vel_vec_handles) >= 3:
        vel_handles += [
            vel_vec_handles[0],
            vel_vec_handles[1],
            vel_vec_handles[2],
        ]
        vel_handles[-3].set_label("v1")
        vel_handles[-2].set_label("v2")
        vel_handles[-1].set_label("v3")
    ax2.legend(handles=vel_handles, loc="upper right")

    plt.tight_layout()
    savefig(fig, outdir / filename, show=show)
