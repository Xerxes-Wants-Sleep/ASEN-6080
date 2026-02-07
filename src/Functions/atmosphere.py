import numpy as np


def rho_exp(r_mag: float, *, rho0: float, r0: float, H: float) -> float:
    """Exponential density model.

    rho(r) = rho0 * exp(-(r - r0)/H)

    Parameters
    ----------
    r_mag : float
        Radius magnitude (same units as r0, H).
    rho0 : float
        Reference density at r0.
    r0 : float
        Reference radius.
    H : float
        Scale height.
    """
    return float(rho0) * float(np.exp(-(float(r_mag) - float(r0)) / float(H)))


def grad_rho_exp(
    r_vec: np.ndarray,
    *,
    rho0: float,
    r0: float,
    H: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Gradient of the exponential density model w.r.t. Cartesian position.

    Returns
    -------
    grad : (3,) ndarray
        grad[i] = d rho / d r_i
    """
    r = np.asarray(r_vec, dtype=float).reshape(3)
    r_mag = float(np.linalg.norm(r))
    if r_mag < eps:
        return np.zeros(3)

    rho = rho_exp(r_mag, rho0=rho0, r0=r0, H=H)
    drho_drmag = -(rho / float(H))
    return drho_drmag * (r / r_mag)
