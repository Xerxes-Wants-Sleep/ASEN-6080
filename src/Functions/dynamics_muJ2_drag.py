import numpy as np

from .jacobians import (
    accel_wJ2J3,
    dadr_wJ2J3,
    da_dparams_wJ2J3,
    drag_partials_exp,
)


def _skew(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [w]x such that [w]x r = w × r."""
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz,  wy],
                     [wz,  0.0, -wx],
                     [-wy, wx,  0.0]], dtype=float)


def f_muJ2_drag(t: float, x: np.ndarray, const: dict) -> np.ndarray:
    """
    Project 1 dynamics for (nominally) 18-state:
        x = [r(3), v(3), mu, J2, Cd, Rs_101(3), Rs_337(3), Rs_394(3)]

    Assumptions:
      - mu, J2, Cd are CONSTANT in the dynamics (dot = 0).
      - Station position states Rs_* are stored in INERTIAL and rotate with Earth:
            d(Rs)/dt = omega x Rs
      - Spacecraft acceleration = gravity(mu,J2) + drag(Cd, exp atmosphere).
      - Optional rotating-atmosphere model for drag:
            v_rel = v - omega_atm x r   (controlled by const flag)
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    r = x[0:3]
    v = x[3:6]
    mu = float(x[6])
    J2 = float(x[7])
    Cd = float(x[8])

    # ---- constants ----
    Re = float(const.get("Re", 6378136.3))  # meters default
    drag_on = bool(const.get("drag", True))
    atmosphere_rotates = bool(const.get("atmosphere_rotates", True))

    # Earth rotation vector (rad/s)
    omega_vec = np.asarray(const.get("omega_vec", [0.0, 0.0, 7.2921158553e-5]), dtype=float).reshape(3)

    # Exponential density params (meters-based defaults from project)
    rho0 = float(const.get("rho0", 3.614e-13))
    r0 = float(const.get("r0", 700e3 + Re))
    H = float(const.get("H", 88667.0))

    # Spacecraft properties
    area = float(const.get("A", 3.0))
    mass = float(const.get("m", 970.0))

    # ---- acceleration ----
    a_grav = accel_wJ2J3(r, mu=mu, J2=J2, J3=0.0, Re=Re, j2=True, j3=False)

    if drag_on:
        a_drag, _, _, _ = drag_partials_exp(
            r=r,
            v=v,
            Cd=Cd,
            area=area,
            mass=mass,
            rho0=rho0,
            r0=r0,
            H=H,
            omega_vec=omega_vec,
            atmosphere_rotates=atmosphere_rotates,
        )
    else:
        a_drag = np.zeros(3, dtype=float)

    a = a_grav + a_drag

    # ---- build xdot ----
    xdot = np.zeros_like(x)
    xdot[0:3] = v
    xdot[3:6] = a
    # xdot[6:9] = 0 implicitly

    # ---- station kinematics (ONLY the three 3-vectors) ----
    # Standard Project 1 layout: stations at indices 9:18
    # If you ever change number of stations, adjust these blocks.
    if x.size >= 18:
        for base in (9, 12, 15):
            Rs_i = x[base:base+3]
            xdot[base:base+3] = np.cross(omega_vec, Rs_i)

    return xdot


def A_muJ2_drag(t: float, x: np.ndarray, const: dict) -> np.ndarray:
    """
    Variational matrix A = df/dx for Project 1 dynamics.

    State ordering (nominally 18):
        x = [r(3), v(3), mu, J2, Cd, Rs_101(3), Rs_337(3), Rs_394(3)]

    Notes:
      - Includes station rotation block: d(Rs_dot)/d(Rs) = [omega]x
      - mu, J2, Cd have zero dynamics (rows are zero), but acceleration depends on them.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n < 9:
        raise ValueError(f"State must have at least 9 elements (r,v,mu,J2,Cd). Got n={n}.")

    r = x[0:3]
    v = x[3:6]
    mu = float(x[6])
    J2 = float(x[7])
    Cd = float(x[8])

    # ---- constants ----
    Re = float(const.get("Re", 6378136.3))
    drag_on = bool(const.get("drag", True))
    atmosphere_rotates = bool(const.get("atmosphere_rotates", True))
    omega_vec = np.asarray(const.get("omega_vec", [0.0, 0.0, 7.2921158553e-5]), dtype=float).reshape(3)

    rho0 = float(const.get("rho0", 3.614e-13))
    r0 = float(const.get("r0", 700e3 + Re))
    H = float(const.get("H", 88667.0))
    area = float(const.get("A", 3.0))
    mass = float(const.get("m", 970.0))

    # ---- allocate A ----
    A = np.zeros((n, n), dtype=float)

    # dr/dt = v
    A[0:3, 3:6] = np.eye(3)

    # ---- gravity partials (J3 off) ----
    G_grav = dadr_wJ2J3(r, mu=mu, J2=J2, J3=0.0, Re=Re, j2=True, j3=False)  # 3x3
    da_dmu, da_dJ2, _ = da_dparams_wJ2J3(r, mu=mu, J2=J2, J3=0.0, Re=Re, j2=True, j3=False)  # (3,)

    # ---- drag partials ----
    if drag_on:
        _, dadr_drag, dadv_drag, dadCd = drag_partials_exp(
            r=r,
            v=v,
            Cd=Cd,
            area=area,
            mass=mass,
            rho0=rho0,
            r0=r0,
            H=H,
            omega_vec=omega_vec,
            atmosphere_rotates=atmosphere_rotates,
        )
    else:
        dadr_drag = np.zeros((3, 3), dtype=float)
        dadv_drag = np.zeros((3, 3), dtype=float)
        dadCd = np.zeros(3, dtype=float)

    # dv/dt partials
    A[3:6, 0:3] = G_grav + dadr_drag
    A[3:6, 3:6] = dadv_drag
    A[3:6, 6] = da_dmu
    A[3:6, 7] = da_dJ2
    A[3:6, 8] = dadCd

    # ---- station rotation Jacobian blocks ----
    # Rs_dot = omega x Rs  =>  d(Rs_dot)/d(Rs) = [omega]x
    if n >= 18:
        A_gs = _skew(omega_vec)
        for base in (9, 12, 15):
            A[base:base+3, base:base+3] = A_gs

    return A
