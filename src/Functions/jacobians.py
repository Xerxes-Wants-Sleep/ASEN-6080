import numpy as np
from .atmosphere import rho_exp, grad_rho_exp  # For drag stuff

def skew(w: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix such that skew(w) @ v == w x v."""
    w = np.asarray(w, dtype=float).reshape(3)
    wx, wy, wz = w
    return np.array(
        [[0.0, -wz, wy],
         [wz, 0.0, -wx],
         [-wy, wx, 0.0]],
        dtype=float,
    )

def accel_wJ2J3(r, mu, J2, J3, Re=6378, j2=True, j3=True):
    """
    r is a pos vel array
    Acceleration a(r) = a_mu + a_J2 + a_J3 in Cartesian
    j2 j3 = togglable
    """ 

    x, y, z = r
    r2 = x*x + y*y + z*z
    rmag = np.sqrt(r2)

    # Central term
    a = -mu * r / (rmag**3)

    # J2 
    if j2:
        z2 = z*z
        u2 = np.array([
            x * (r2 - 5.0*z2),
            y * (r2 - 5.0*z2),
            z * (3.0*r2 - 5.0*z2)
        ])
        a -= (3.0*mu*J2*(Re**2) / (2.0*(rmag**7))) * u2

    # J3
    if j3:
        z2 = z*z
        r4 = r2*r2
        z4 = z2*z2
        C = 7.0*z2 - 3.0*r2
        D = 3.0*r4 - 30.0*r2*z2 + 35.0*z4
        w = np.array([
            5.0*x*z*C,
            5.0*y*z*C,
            D
        ])
        a += (mu*J3*(Re**3) / (2.0*(rmag**9))) * w

    return a


def dadr_wJ2J3(r, mu, J2, J3, Re=6378, j2=True, j3=True):
    """
    Gravity-gradient matrix G = da/dr (3x3) for a = a_mu + a_J2 + a_J3.

    Returns
    G : array, shape (3,3)
    """
    x, y, z = r
    r2 = x*x + y*y + z*z
    rmag = np.sqrt(r2)
    I = np.eye(3)
    rrT = np.outer(r, r)

    # Central term
    G = (mu / (rmag**5)) * (3.0*rrT - r2*I)

    # J2 
    if j2:
        z2 = z*z
        u2 = np.array([
            x * (r2 - 5.0*z2),
            y * (r2 - 5.0*z2),
            z * (3.0*r2 - 5.0*z2)
        ])

        
        M2 = np.array([
            [(r2 - 5.0*z2) + 2.0*x*x,  2.0*x*y,              -8.0*x*z],
            [2.0*x*y,                  (r2 - 5.0*z2) + 2.0*y*y, -8.0*y*z],
            [6.0*x*z,                  6.0*y*z,              (3.0*r2 - 5.0*z2) - 4.0*z2]
        ])

        coeff = (3.0*mu*J2*(Re**2)) / (2.0*(rmag**7))
        G -= coeff * (M2 - (7.0/r2)*np.outer(u2, r))

    # J3
    if j3:
        z2 = z*z
        r4 = r2*r2
        z4 = z2*z2

        C = 7.0*z2 - 3.0*r2
        D = 3.0*r4 - 30.0*r2*z2 + 35.0*z4

        w = np.array([
            5.0*x*z*C,
            5.0*y*z*C,
            D
        ])

        M3 = np.array([
            [5.0*z*(C - 6.0*x*x),   -30.0*x*y*z,        5.0*x*(C + 8.0*z2)],
            [-30.0*x*y*z,           5.0*z*(C - 6.0*y*y), 5.0*y*(C + 8.0*z2)],
            [12.0*x*(r2 - 5.0*z2),  12.0*y*(r2 - 5.0*z2), 16.0*z*(5.0*z2 - 3.0*r2)]
        ])

        coeff = (mu*J3*(Re**3)) / (2.0*(rmag**9))
        G += coeff * (M3 - (9.0/r2)*np.outer(w, r))

    return G


def da_dparams_wJ2J3(r, mu, J2, J3, Re=6378, j2=True, j3=True):
    """
    Parameter partials: da WRT mu, j2, j3.
    """
    a = accel_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)
    
    da_dmu = a / mu

    # da/dj2
    if j2:
        x, y, z = r
        r2 = x*x + y*y + z*z
        rmag = np.sqrt(r2)
        z2 = z*z
        u2 = np.array([
            x * (r2 - 5.0*z2),
            y * (r2 - 5.0*z2),
            z * (3.0*r2 - 5.0*z2)
        ])
        da_dJ2 = -(3.0*mu*(Re**2) / (2.0*(rmag**7))) * u2
    else:
        da_dJ2 = np.zeros(3)

    # da/dj3
    if j3:
        x, y, z = r
        r2 = x*x + y*y + z*z
        rmag = np.sqrt(r2)
        z2 = z*z
        r4 = r2*r2
        z4 = z2*z2
        C = 7.0*z2 - 3.0*r2
        D = 3.0*r4 - 30.0*r2*z2 + 35.0*z4
        w = np.array([5.0*x*z*C, 5.0*y*z*C, D])
        da_dJ3 = (mu*(Re**3) / (2.0*(rmag**9))) * w
    else:
        da_dJ3 = np.zeros(3)

    return da_dmu, da_dJ2, da_dJ3


def state_builder(state9, Re=6378, j2=True, j3=True):
    """
    Build the 9x9 A-matrix for augmented state:
      X = [r(3), v(3), mu, J2, J3].

    Returns
    -------
    A : 9x9 array of STM
    """
    state9 = np.asarray(state9, dtype=float).reshape(-1)

    r = state9[0:3]
    v = state9[3:6]  # not needed for Jacobian 
    mu = state9[6]
    J2 = state9[7]
    J3 = state9[8]

    G = dadr_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)         # 3x3
    da_dmu, da_dJ2, da_dJ3 = da_dparams_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)

    A = np.zeros((9, 9), dtype=float)

    # dr/dt = v
    A[0:3, 3:6] = np.eye(3)

    # dv/dt = a
    A[3:6, 0:3] = G
    # dv/dt partials wrt params
    A[3:6, 6] = da_dmu
    A[3:6, 7] = da_dJ2
    A[3:6, 8] = da_dJ3
    return A


def orbit_propagator(t, X, mu, J2, J3, Re=6378, j2=True, j3=True):
    """
    6-state dynamics: X = [r(3), v(3)].
    """
    r = np.asarray(X[0:3], dtype=float)
    v = np.asarray(X[3:6], dtype=float)
    a = accel_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)
    return np.hstack((v, a))


def orbit_propagator_aug9(t, X9, Re=6378, j2=True, j3=True):
    """
    9-state dynamics: X9 = [r(3), v(3), mu, J2, J3]
    """
    X9 = np.asarray(X9, dtype=float).reshape(-1)
    r = X9[0:3]
    v = X9[3:6]
    mu, J2, J3 = X9[6], X9[7], X9[8]
    a = accel_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)
    # parameters are constant -> zeros
    return np.hstack((v, a, 0.0, 0.0, 0.0))


def stm(t, state9: np.ndarray, phi: np.ndarray, rows_col_to_remove: np.ndarray, Re=6378, j2=True, j3=True):

    '''Build the 6x6 A-matrix directly (rows_col_to_remove is always [6,7,8])
       rather than building 9x9 and calling np.delete on every integrator step.'''
    state9 = np.asarray(state9, dtype=float).reshape(-1)
    mu = state9[6]
    J2 = state9[7]
    J3 = state9[8]
    r  = state9[0:3]
    v  = state9[3:6]

    # Build 6x6 A directly - no 9x9 allocation, no np.delete
    G = dadr_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)   # 3x3 gravity gradient
    a = accel_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)  # 3-vec acceleration

    A_new = np.zeros((6, 6), dtype=float)
    A_new[0:3, 3:6] = np.eye(3)   # dr/dt = v
    A_new[3:6, 0:3] = G           # dv/dt = G * dr

    phi_dot = A_new @ phi
    dxdt = np.hstack((v, a, 0.0, 0.0, 0.0))

    return np.hstack((dxdt, phi_dot.reshape(-1)))


##### PROJ 1 Addons #####

def stm_generic(
    t: float,
    y: np.ndarray,
    n: int,
    f,
    A,
) -> np.ndarray:
    """Combined state+STM dynamics for state vector size n

    Parameters
    ----------
    t : float
        Current time.
    y : (n + n*n,) ndarray
        Stacked vector [x; Phi_flat] where Phi is (n,n).
    n : int
        State dimension.
    f : callable
        Dynamics function f(t, x) -> xdot, shape (n,).
    A : callable
        Variational matrix function A(t, x) -> d f / d x, shape (n,n).

    Returns
    -------
    dy : (n + n*n,) ndarray
        Stacked derivative [xdot; Phidot_flat].
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size != n + n * n:
        raise ValueError(f"stm_generic expected y of length {n + n*n}, got {y.size}.")

    x = y[:n]
    Phi = y[n:].reshape(n, n)

    xdot = np.asarray(f(t, x), dtype=float).reshape(n)
    A_mat = np.asarray(A(t, x), dtype=float)
    if A_mat.shape != (n, n):
        raise ValueError(f"A(t,x) must be ({n},{n}), got {A_mat.shape}.")

    Phidot = A_mat @ Phi
    return np.hstack((xdot, Phidot.reshape(-1)))




def accel_drag_exp(
    r: np.ndarray,
    v: np.ndarray,
    Cd: float,
    area: float,
    mass: float,
    rho0: float,
    r0: float,
    H: float,
    omega_vec=None,
    atmosphere_rotates: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Drag acceleration for an exponential density model.

    All inputs must be in consistent units (typically meters, seconds, kg).

    a_drag = -0.5 * rho * Cd * (A/m) * ||v_rel|| * v_rel
    where v_rel = v - omega x r if atmosphere_rotates=True.
    """
    from .atmosphere import rho_exp

    r = np.asarray(r, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    omega = np.zeros(3) if omega_vec is None else np.asarray(omega_vec, dtype=float).reshape(3)

    rmag = float(np.linalg.norm(r))
    rho = float(rho_exp(rmag, rho0=rho0, r0=r0, H=H))

    if atmosphere_rotates:
        v_rel = v - np.cross(omega, r)
    else:
        v_rel = v

    vrel_mag = float(np.linalg.norm(v_rel))
    if vrel_mag < eps or rho == 0.0:
        return np.zeros(3)

    k = 0.5 * Cd * area / mass
    return -k * rho * vrel_mag * v_rel


def drag_partials_exp(
    r: np.ndarray,
    v: np.ndarray,
    Cd: float,
    area: float,
    mass: float,
    rho0: float,
    r0: float,
    H: float,
    omega_vec=None,
    atmosphere_rotates: bool = True,
    eps: float = 1e-12,
):
    """Drag accel and partials for exponential atmosphere.

    Returns
    -------
    a_drag : (3,) ndarray
    dadr   : (3,3) ndarray
        Partial of drag acceleration wrt position r.
    dadv   : (3,3) ndarray
        Partial of drag acceleration wrt velocity v.
    dadCd  : (3,) ndarray
        Partial of drag acceleration wrt Cd.
    """
    from .atmosphere import rho_exp, grad_rho_exp

    r = np.asarray(r, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    omega = np.zeros(3) if omega_vec is None else np.asarray(omega_vec, dtype=float).reshape(3)

    rmag = float(np.linalg.norm(r))
    rho = float(rho_exp(rmag, rho0=rho0, r0=r0, H=H))
    grad_rho = np.asarray(grad_rho_exp(r, rho0=rho0, r0=r0, H=H, eps=eps), dtype=float).reshape(3)

    if atmosphere_rotates:
        B = -skew(omega)  # du/dr (u = v - omega x r)
        u = v - np.cross(omega, r)
    else:
        B = np.zeros((3, 3))
        u = v

    vrel = float(np.linalg.norm(u))
    if vrel < eps or rho == 0.0:
        a = np.zeros(3)
        return a, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3)

    k = 0.5 * Cd * area / mass
    I = np.eye(3)

    # g(u) = ||u|| * u
    g = vrel * u
    M = (np.outer(u, u) / vrel) + vrel * I  # dg/du

    # a = -k * rho * g
    a = -k * rho * g

    # da/dv = -k * rho * dg/du * du/dv, with du/dv = I
    dadv = -k * rho * M

    # da/dr = -k * ( g ⊗ grad_rho + rho * dg/du * du/dr )
    dgdr = M @ B
    dadr = -k * (np.outer(g, grad_rho) + rho * dgdr)

    # da/dCd
    if abs(Cd) > eps:
        dadCd = a / Cd
    else:
        # limit as Cd->0 (rare, but avoids division-by-zero)
        dadCd = -0.5 * rho * (area / mass) * g

    return a, dadr, dadv, dadCd