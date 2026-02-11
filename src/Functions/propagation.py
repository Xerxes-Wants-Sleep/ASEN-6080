from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from .jacobians import stm_generic


StateFn = Callable[[float, np.ndarray], np.ndarray]
AMatrixFn = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PropSettings:
    """Numerical integration settings for orbit + STM propagation."""

    rtol: float = 1e-10
    atol: float = 1e-10
    method: str = "DOP853"


def propagate_x_phi_step(
    *,
    x0: np.ndarray,
    t0: float,
    t1: float,
    f: StateFn,
    A: AMatrixFn,
    settings: PropSettings | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate state and STM from t0 to t1.

    Returns
    -------
    x1 : (n,) ndarray
    Phi_10 : (n,n) ndarray
        Phi(t1, t0)
    """
    settings = settings or PropSettings()

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    n = x0.size

    Phi0 = np.eye(n)
    y0 = np.hstack((x0, Phi0.reshape(-1)))

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return stm_generic(t, y, n, f, A)

    sol = solve_ivp(
        rhs,
        (float(t0), float(t1)),
        y0,
        t_eval=[float(t1)],
        rtol=settings.rtol,
        atol=settings.atol,
        method=settings.method,
    )
    if not sol.success:
        raise RuntimeError(f"propagate_x_phi_step failed: {sol.message}")

    yf = sol.y[:, -1]
    x1 = yf[:n]
    Phi_10 = yf[n:].reshape(n, n)
    return x1, Phi_10


def propagate_x_phi_history(
    *,
    x0: np.ndarray,
    t_eval: np.ndarray,
    f: StateFn,
    A: AMatrixFn,
    settings: PropSettings | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate state and STM from t_eval[0] to each time in t_eval.

    Parameters
    ----------
    x0 : (n,) ndarray
        Initial state at t_eval[0].
    t_eval : (m,) array_like
        Monotonically increasing times.

    Returns
    -------
    X_hist : (m,n) ndarray
    Phi_i0 : (m,n,n) ndarray
        Phi(t_i, t_0) for each i.
    """
    settings = settings or PropSettings()

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    n = x0.size

    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    if t_eval.size < 1:
        raise ValueError("t_eval must have at least one time.")
    if np.any(np.diff(t_eval) < 0):
        raise ValueError("t_eval must be monotonically increasing.")

    t0 = float(t_eval[0])
    tf = float(t_eval[-1])

    Phi0 = np.eye(n)
    y0 = np.hstack((x0, Phi0.reshape(-1)))

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return stm_generic(t, y, n, f, A)

    sol = solve_ivp(
        rhs,
        (t0, tf),
        y0,
        t_eval=t_eval,
        rtol=settings.rtol,
        atol=settings.atol,
        method=settings.method,
    )
    if not sol.success:
        raise RuntimeError(f"propagate_x_phi_history failed: {sol.message}")

    Y = sol.y.T  # (m, n+n^2)
    X_hist = Y[:, :n]
    Phi_i0 = Y[:, n:].reshape(t_eval.size, n, n)
    return X_hist, Phi_i0


def phi_i0_to_phi_steps(Phi_i0: np.ndarray, *, eps: float = 1e-14) -> np.ndarray:
    """Convert Phi(t_i,t0) history to step STMs Phi(t_i,t_{i-1}).

    Parameters
    ----------
    Phi_i0 : (m,n,n) ndarray

    Returns
    -------
    Phi_step : (m,n,n) ndarray
        Phi_step[0] = I
        Phi_step[i] = Phi(t_i, t_{i-1}) for i>=1
    """
    Phi_i0 = np.asarray(Phi_i0, dtype=float)
    if Phi_i0.ndim != 3:
        raise ValueError("Phi_i0 must have shape (m,n,n).")
    m, n, n2 = Phi_i0.shape
    if n != n2:
        raise ValueError("Phi_i0 must have shape (m,n,n).")

    Phi_step = np.zeros_like(Phi_i0)
    Phi_step[0] = np.eye(n)

    # Phi(t_i,t_{i-1}) = Phi(t_i,t0) * inv(Phi(t_{i-1},t0))
    # Use solve for numerical stability.
    for i in range(1, m):
        prev = Phi_i0[i - 1]
        # Right-multiply by inv(prev) without forming inv:
        #   M = Phi_i0[i] @ inv(prev)
        # => M^T = solve(prev^T, Phi_i0[i]^T)
        Phi_step[i] = np.linalg.solve(prev.T, Phi_i0[i].T).T

    return Phi_step
