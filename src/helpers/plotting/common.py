from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def as_hours(t_seconds: np.ndarray) -> np.ndarray:
    t_seconds = np.asarray(t_seconds, dtype=float).reshape(-1)
    return t_seconds / 3600.0


def rms_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(x * x)))


def station_masks(station_names: list[str]) -> dict[str, np.ndarray]:
    """Return boolean mask per station name."""
    uniq = list(dict.fromkeys(station_names))  # preserve order
    masks = {}
    arr = np.array(station_names, dtype=object)
    for s in uniq:
        masks[s] = (arr == s)
    return masks


def savefig(fig: plt.Figure, outpath: Path, show: bool = False, dpi: int = 300) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
