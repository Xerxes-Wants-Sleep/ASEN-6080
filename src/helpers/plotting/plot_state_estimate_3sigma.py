from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import savefig


def _get_state_and_cov(out: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_hr = np.asarray(out["t_meas"], dtype=float) / 3600.0
    xhat_key = "xhat_meas" if "xhat_meas" in out else "Xhat_meas"
    phat_key = "P_meas" if "P_meas" in out else "Phat_meas"

    xhat = np.asarray(out[xhat_key], dtype=float)
    P = np.asarray(out[phat_key], dtype=float)
    if xhat.ndim != 2:
        raise ValueError(f"Expected 2D state history, got shape {xhat.shape}")
    if P.ndim != 3 or P.shape[1] != xhat.shape[1] or P.shape[2] != xhat.shape[1]:
        raise ValueError(f"Expected covariance history shape (N,n,n), got {P.shape}")

    return t_hr, xhat, P


def make_all_state_3sigma_envelope_plot(out: dict, outdir: Path, show: bool = False) -> None:
    """
    Plot estimated states with +/-3 Sigma covariance envelopes for all states.
    Bounds are computed from covariance diagonals:
        3 Sigma_i(t) = 3 * sqrt(P_ii(t)).
    """
    t_hr, xhat, P = _get_state_and_cov(out)
    sig3 = 3.0 * np.sqrt(np.maximum(np.diagonal(P, axis1=1, axis2=2), 0.0))
    n = xhat.shape[1]

    labels_default = [
        "X [km]",
        "Y [km]",
        "Z [km]",
        "Vx [km/s]",
        "Vy [km/s]",
        "Vz [km/s]",
        "Cr [-]",
    ]
    labels = labels_default[:n] + [f"State {i + 1}" for i in range(len(labels_default), n)]

    fig, axs = plt.subplots(n, 1, figsize=(10, max(6, int(1.75 * n))), sharex=True)
    if n == 1:
        axs = [axs]

    for i in range(n):
        axs[i].plot(t_hr, xhat[:, i], "k", linewidth=1.0, label="Estimate")
        axs[i].plot(t_hr, xhat[:, i] + sig3[:, i], "r--", linewidth=1.0, label="+3 Sigma")
        axs[i].plot(t_hr, xhat[:, i] - sig3[:, i], "r--", linewidth=1.0, label="-3 Sigma")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc="best", fontsize=8)

    axs[-1].set_xlabel("Time [Hours]")
    fig.suptitle("Estimated States With +/-3 Sigma Covariance Envelopes")
    savefig(fig, outdir / "state_estimates_all_3sigma_envelopes.png", show=show)


def make_cr_3sigma_plot(out: dict, outdir: Path, show: bool = False, cr_index: int = 6) -> None:
    """
    Plot Cr estimate with +/-3 Sigma envelope.
    """
    t_hr, xhat, P = _get_state_and_cov(out)
    if xhat.shape[1] <= int(cr_index):
        raise ValueError(f"Requested Cr index {cr_index} but state only has {xhat.shape[1]} elements.")

    cr_index = int(cr_index)
    sig3_cr = 3.0 * np.sqrt(np.maximum(P[:, cr_index, cr_index], 0.0))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_hr, xhat[:, cr_index], "k", linewidth=1.2, label="Cr Estimate")
    ax.plot(t_hr, xhat[:, cr_index] + sig3_cr, "r--", linewidth=1.0, label="+3 Sigma")
    ax.plot(t_hr, xhat[:, cr_index] - sig3_cr, "r--", linewidth=1.0, label="-3 Sigma")
    ax.set_xlabel("Time [Hours]")
    ax.set_ylabel("Cr [-]")
    ax.set_title("Cr Estimate With +/-3 Sigma Covariance Envelope")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    savefig(fig, outdir / "cr_estimate_3sigma_envelope.png", show=show)

