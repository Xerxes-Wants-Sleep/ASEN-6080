from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .common import savefig


def make_final_state_estimate_plot(
    out: dict,
    outdir: Path,
    show: bool = False,
    filename: str = "state_estimates_all_3sigma_envelopes.png",
) -> None:
    """
    Plot state estimate time histories with +/-3 sigma envelopes.

    Expected keys in `out`:
      - xhat_meas or Xhat_meas: (m, n)
      - P_meas or Phat_meas: (m, n, n)
    """
    xhat_key = "xhat_meas" if "xhat_meas" in out else "Xhat_meas"
    phat_key = "P_meas" if "P_meas" in out else "Phat_meas"

    X = np.asarray(out[xhat_key], dtype=float)
    P = np.asarray(out[phat_key], dtype=float)
    if X.ndim != 2 or P.ndim != 3 or X.shape[0] == 0:
        raise ValueError("Expected non-empty xhat_meas/Xhat_meas and P_meas/Phat_meas histories.")

    t_hr = np.asarray(out["t_meas"], dtype=float).reshape(-1) / 3600.0
    sig3 = 3.0 * np.sqrt(np.maximum(np.diagonal(P, axis1=1, axis2=2), 0.0))

    n = X.shape[1]
    labels = ["X [km]", "Y [km]", "Z [km]", "Vx [km/s]", "Vy [km/s]", "Vz [km/s]", "Cr [-]"]
    while len(labels) < n:
        labels.append(f"State {len(labels) + 1} [-]")

    fig, axs = plt.subplots(n, 1, figsize=(10, max(5, 1.85 * n)), sharex=True)
    if n == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        ax.plot(t_hr, X[:, i], "k", linewidth=1.0, label="Estimate")
        ax.plot(t_hr, X[:, i] + sig3[:, i], "r--", linewidth=1.0, label="+3 Sigma")
        ax.plot(t_hr, X[:, i] - sig3[:, i], "r--", linewidth=1.0, label="-3 Sigma")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    axs[-1].set_xlabel("Time [Hours]")
    fig.suptitle("Estimated States With +/-3 Sigma Covariance Envelopes")
    savefig(fig, outdir / filename, show=show)
