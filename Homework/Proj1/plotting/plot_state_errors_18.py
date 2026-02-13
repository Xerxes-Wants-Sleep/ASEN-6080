from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.helpers.plotting.common import as_hours, savefig


def make_state_errors_18_plot(
    t_meas: np.ndarray,
    state_err: np.ndarray,
    outdir: Path,
    *,
    show: bool = False,
    title: str = "State Errors (18-state)",
    filename: str = "state_errors_18.png",
) -> None:
    """
    Plot 18-state error time histories in a 6x3 grid.

    state_err is expected to be (m, 18) with units:
      r (m), v (m/s), mu (m^3/s^2), J2 (unitless), Cd (unitless),
      station positions (m).
    """
    t_meas = np.asarray(t_meas, dtype=float).reshape(-1)
    e = np.asarray(state_err, dtype=float)
    if e.ndim != 2 or e.shape[1] < 18:
        raise ValueError(f"state_err must have shape (m, 18); got {e.shape}")

    t_hr = as_hours(t_meas)

    labels = [
        "x [m]", "y [m]", "z [m]",
        "vx [m/s]", "vy [m/s]", "vz [m/s]",
        "mu [m^3/s^2]", "J2 [-]", "Cd [-]",
        "Rs_101 x [m]", "Rs_101 y [m]", "Rs_101 z [m]",
        "Rs_337 x [m]", "Rs_337 y [m]", "Rs_337 z [m]",
        "Rs_394 x [m]", "Rs_394 y [m]", "Rs_394 z [m]",
    ]

    fig, axes = plt.subplots(6, 3, figsize=(14, 10), sharex=True)
    axes = axes.reshape(6, 3)

    for i in range(18):
        r = i // 3
        c = i % 3
        ax = axes[r, c]
        ax.plot(t_hr, e[:, i], ".", markersize=2)
        ax.set_ylabel(labels[i])
        if r == 5:
            ax.set_xlabel("Time [hours]")

    fig.suptitle(title)
    savefig(fig, outdir / filename, show=show)
