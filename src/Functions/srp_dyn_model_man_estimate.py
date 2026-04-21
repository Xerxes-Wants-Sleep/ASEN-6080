"""
maneuver_estimation.py
======================
All functions needed to add delta-v maneuver estimation to the 9-state IEKF.

State layout (9-state, no Cr):
    X = [x, y, z,  xdot, ydot, zdot,  dvx, dvy, dvz]
         0  1  2   3     4     5       6    7    8

Δv is constant in continuous time (dΔv/dt = 0).
It enters velocity instantaneously at the maneuver time t_man
via the discrete jump applied inside the filter.
"""

from __future__ import annotations
import numpy as np
from .jacobians import dadr_2body, third_body_accel_partials, cannonball_SRP
from .propagation import PropSettings, propagate_x_phi_step


# ---------------------------------------------------------------------------
# 1.  Continuous-time Jacobian  A  (9x9)
# ---------------------------------------------------------------------------

def build_A_9state(
    r_sc: np.ndarray,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    Cr: float,
    area: float,
    mass: float,
    mu_earth: float,
    mu_sun: float,
    solar_flux_1au: float = 1357.0,
    c: float = 299792458.0,
    AU_m: float = 149597870700.0,
) -> np.ndarray:
    """
    Continuous-time Jacobian A for the 9-state system.

    State:  X = [r(0:3), v(3:6), Δv(6:9)]
    Cr is a fixed parameter, NOT part of the state.

    Structure:
         r       v      Δv
    r  [ 0₃    I₃     0₃  ]
    v  [ G     0₃     0₃  ]   <- G = gravity gradient (2body+3body+SRP)
    Δv [ 0₃    0₃     0₃  ]   <- Δv is constant in continuous time

    The Δv columns are all zero here.  The identity coupling
    v ← Δv appears only in the discrete jump matrix at t_man.

    Parameters
    ----------
    r_sc, r_earth, r_sun : (3,) arrays in km
    Cr   : fixed reflectivity coefficient (not estimated)
    area : m^2
    mass : kg
    mu_earth, mu_sun : km^3/s^2
    solar_flux_1au, c, AU_m : SRP constants in SI

    Returns
    -------
    A : (9, 9) ndarray   [units: 1/s or 1/s^2 as appropriate]
    """
    r_sc    = np.asarray(r_sc,    dtype=float).reshape(3)
    r_earth = np.asarray(r_earth, dtype=float).reshape(3)
    r_sun   = np.asarray(r_sun,   dtype=float).reshape(3)

    G_2b        = dadr_2body(r_sc - r_earth, mu_earth)
    _, G_3b     = third_body_accel_partials(r_sc, r_earth, r_sun, mu_sun)
    _, G_srp, _ = cannonball_SRP(r_sc, r_sun, Cr, area, mass,
                                  solar_flux_1au, c, AU_m)

    G = G_2b + G_3b + G_srp          # (3,3)  total gravity gradient

    A = np.zeros((9, 9), dtype=float)
    A[0:3, 3:6] = np.eye(3)           # dr/dt = v
    A[3:6, 0:3] = G                   # dv/dt depends on r
    # rows 6:9 are zero  — Δv is constant in continuous time
    # cols 6:9 are zero  — Δv has no continuous effect on r or v

    return A


# ---------------------------------------------------------------------------
# 2.  State derivative  (9-state, returns dX/dt, no STM)
# ---------------------------------------------------------------------------

def state_deriv_9state(
    t: float,
    X: np.ndarray,
    pConst,
    scConst,
    earth_state_func,
    sun_state_func,
) -> np.ndarray:
    """
    Time derivative of the 9-state vector.
    Δv states are frozen (dΔv/dt = 0) in continuous time.
    """
    X = np.asarray(X, dtype=float).reshape(9)
    r_sc = X[0:3]
    v_sc = X[3:6]
    # X[6:9] is Δv — not used in continuous dynamics

    r_earth, _ = earth_state_func(t)
    r_sun,   _ = sun_state_func(t)

    r = r_sc - r_earth
    rmag = np.linalg.norm(r)
    a_mu = -pConst.mu_earth * r / rmag**3

    d_sc    = r_sun - r_sc
    d_earth = r_sun - r_earth
    a_3b = pConst.mu_sun * (
        d_sc    / np.linalg.norm(d_sc)**3 -
        d_earth / np.linalg.norm(d_earth)**3
    )

    a_srp, _, _ = cannonball_SRP(
        r_sc=r_sc, r_sun=r_sun,
        Cr=scConst.Cr_fixed,
        area=scConst.area, mass=scConst.mass,
        solar_flux_1au=scConst.solar_flux_1au,
        c=scConst.c, AU_m=scConst.AU_m,
    )

    dX = np.zeros(9, dtype=float)
    dX[0:3] = v_sc
    dX[3:6] = a_mu + a_3b + a_srp
    dX[6:9] = 0.0          # Δv constant
    return dX


# ---------------------------------------------------------------------------
# 3.  Combined state + STM derivative  (for ODE integration, 9+81 = 90 elem)
# ---------------------------------------------------------------------------

def stm_deriv_9state(
    t: float,
    XPhi: np.ndarray,
    pConst,
    scConst,
    earth_state_func,
    sun_state_func,
) -> np.ndarray:
    """
    ODE right-hand side for the stacked [X(9); Phi_flat(81)] vector.

    This is what you pass to propagate_x_phi_step (or your ODE integrator)
    for normal propagation steps where no maneuver is occurring.

    Note: the Δv columns of Phi stay zero throughout normal propagation
    because A[:,6:9] = 0.  They are only filled in by apply_maneuver_jump.
    """
    XPhi = np.asarray(XPhi, dtype=float).reshape(-1)
    n    = 9
    X    = XPhi[:n]
    Phi  = XPhi[n:].reshape(n, n)

    dX  = state_deriv_9state(t, X, pConst, scConst, earth_state_func, sun_state_func)

    r_sc    = X[0:3]
    r_earth, _ = earth_state_func(t)
    r_sun,   _ = sun_state_func(t)

    A    = build_A_9state(
        r_sc=r_sc, r_earth=r_earth, r_sun=r_sun,
        Cr=scConst.Cr_fixed,
        area=scConst.area, mass=scConst.mass,
        mu_earth=pConst.mu_earth, mu_sun=pConst.mu_sun,
        solar_flux_1au=scConst.solar_flux_1au,
        c=scConst.c, AU_m=scConst.AU_m,
    )
    dPhi = (A @ Phi).reshape(-1)

    return np.hstack((dX, dPhi))


# ---------------------------------------------------------------------------
# 4.  Measurement partials  H  (2x9)
# ---------------------------------------------------------------------------

def H_9state(
    r_sc: np.ndarray,
    v_sc: np.ndarray,
    r_gs: np.ndarray,
    v_gs: np.ndarray,
) -> np.ndarray:
    """
    Measurement Jacobian H for [range, range-rate] w.r.t. the 9-state.

    State:  X = [r_sc(0:3), v_sc(3:6), Δv(6:9)]

    Δv has NO direct algebraic effect on range or range-rate at a given
    instant — its effect propagates through r and v via the STM — so
    columns 6:9 are zero.

    Returns
    -------
    H : (2, 9) ndarray
    """
    r_sc = np.asarray(r_sc, dtype=float).reshape(3)
    v_sc = np.asarray(v_sc, dtype=float).reshape(3)
    r_gs = np.asarray(r_gs, dtype=float).reshape(3)
    v_gs = np.asarray(v_gs, dtype=float).reshape(3)

    dr      = r_sc - r_gs
    dv      = v_sc - v_gs
    rho     = np.linalg.norm(dr)
    rho_hat = dr / rho
    rv      = np.dot(dr, dv)

    # ∂ρ / ∂r_sc  = rho_hat,   ∂ρ / ∂v_sc  = 0
    drange_dr = rho_hat.reshape(1, 3)
    drange_dv = np.zeros((1, 3))

    # ∂ρ̇ / ∂r_sc = (dv - rv/rho² * rho_hat) / rho
    #             = dv/rho - rv/rho³ * dr
    # ∂ρ̇ / ∂v_sc = rho_hat
    drate_dr = ((dv / rho) - (rv / rho**3) * dr).reshape(1, 3)
    drate_dv = rho_hat.reshape(1, 3)

    H = np.zeros((2, 9), dtype=float)
    H[0, 0:3] = drange_dr
    H[0, 3:6] = drange_dv   # zeros, explicit for clarity
    H[1, 0:3] = drate_dr
    H[1, 3:6] = drate_dv
    # H[:, 6:9] = 0  — Δv columns, no direct measurement dependence

    return H


# ---------------------------------------------------------------------------
# 5.  Discrete maneuver jump  (call this BETWEEN integration steps at t_man)
# ---------------------------------------------------------------------------

def apply_maneuver_jump(
    X: np.ndarray,
    Phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the instantaneous Δv kick to state and STM at the maneuver time.

    MUST be called between two integration steps, never inside the integrator.

    What this does
    --------------
    At t_man the velocity receives an instantaneous kick: v+ = v- + Δv.

    The STM sensitivity of [r,v] to Δv is:
        ∂[r,v](t_man) / ∂Δv = [0₃; I₃]

    Propagated forward to any later time t_k by the r,v block of the STM:
        ∂[r,v](t_k) / ∂Δv = Phi_rv(t_k, t_man) @ [0₃; I₃]
                           = Phi_rv[:, 3:6]   (lower 3 cols of the rv block)

    This is exactly the line your friend has:
        Phi_man(1:6, 8:10) = Phi_rv * [zeros(3,3); eye(3)]

    Parameters
    ----------
    X   : (9,)  current state  [r, v, Δv]
    Phi : (9,9) current STM from t0 to t_man

    Returns
    -------
    X_new   : (9,)  state after kick  (v updated, Δv unchanged)
    Phi_new : (9,9) STM with Δv sensitivity columns filled in
    """
    X   = X.copy()
    Phi = Phi.copy()

    # The 6x6 position-velocity block of the STM
    Phi_rv = Phi[0:6, 0:6]

    # Sensitivity of [r,v] at t_man to a change in Δv:
    #   [0₃ₓ₃]   <- position unaffected at the instant of the kick
    #   [I₃ₓ₃]   <- velocity directly += Δv
    sensitivity = np.vstack([np.zeros((3, 3)), np.eye(3)])   # (6,3)

    # Propagate that sensitivity through the rv dynamics
    # (this is a no-op at t_man itself but correct for later STM steps)
    Phi[0:6, 6:9] = Phi_rv @ sensitivity

    # Apply kick to velocity
    X[3:6] += X[6:9]

    return X, Phi
