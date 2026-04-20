import numpy as np

from .jacobians import cannonball_SRP, srp_thirdbody_variational_eq_for6state


def srp_dyn_for6state(
    mu: float,
    mu_i: float,
    r_sc: np.ndarray,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    Cr: float,
    area: float,
    mass: float,
    solar_flux_1au: float = 1357.0,   # W/m^2 at 1 AU
    c: float = 299792458.0,            # m/s
    AU_m: float = 149597870700,        # m
):
    """
    Total acceleration for fixed-Cr 6-state dynamics:
      a = a_2body + a_3body + a_SRP
    """
    r_sc = np.asarray(r_sc, dtype=float).reshape(3,)
    r_earth = np.asarray(r_earth, dtype=float).reshape(3,)
    r_sun = np.asarray(r_sun, dtype=float).reshape(3,)

    r = r_sc - r_earth
    r2 = float(np.dot(r, r))
    rmag = float(np.sqrt(r2))

    # 2-body
    a_mu = -float(mu) * r / (rmag**3)

    # Third-body (Sun)
    r_i_sc = r_sun - r_sc
    r_i_earth = r_sun - r_earth
    a_i = float(mu_i) * (
        r_i_sc / (np.linalg.norm(r_i_sc) ** 3)
        - r_i_earth / (np.linalg.norm(r_i_earth) ** 3)
    )

    # SRP (fixed Cr)
    a_srp, _, _ = cannonball_SRP(
        r_sc=r_sc,
        r_sun=r_sun,
        Cr=float(Cr),
        area=float(area),
        mass=float(mass),
        solar_flux_1au=float(solar_flux_1au),
        c=float(c),
        AU_m=float(AU_m),
    )
    return a_mu + a_i + a_srp


def mu_sun_srp_state_deriv_for6state(
    t: float,
    X: np.ndarray,
    pConst,
    scConst,
    Cr: float,
    earth_state_func,
    sun_state_func,
):
    """
    6-state dynamics with fixed Cr:
      X = [x, y, z, xdot, ydot, zdot]
    """
    X = np.asarray(X, dtype=float).reshape(6,)

    r_sc = X[0:3]
    v_sc = X[3:6]

    r_earth, _ = earth_state_func(float(t))
    r_sun, _ = sun_state_func(float(t))

    a = srp_dyn_for6state(
        mu=pConst.mu_earth,
        mu_i=pConst.mu_sun,
        r_sc=r_sc,
        r_earth=r_earth,
        r_sun=r_sun,
        Cr=float(Cr),
        area=scConst.area,
        mass=scConst.mass,
        solar_flux_1au=scConst.solar_flux_1au,
        c=scConst.c,
        AU_m=scConst.AU_m,
    )

    dX = np.zeros(6, dtype=float)
    dX[0:3] = v_sc
    dX[3:6] = a
    return dX


def mu_sun_srp_stm_deriv_for6state(
    t: float,
    XPhi: np.ndarray,
    pConst,
    scConst,
    Cr: float,
    earth_state_func,
    sun_state_func,
):
    """
    Combined 6-state + 6x6 STM derivative for fixed-Cr runs.

      XPhi = [X(6); Phi(36)].
    """
    XPhi = np.asarray(XPhi, dtype=float).reshape(-1)

    n = 6
    X = XPhi[:n]
    Phi = XPhi[n:].reshape(n, n)

    dX = mu_sun_srp_state_deriv_for6state(
        t=t,
        X=X,
        pConst=pConst,
        scConst=scConst,
        Cr=float(Cr),
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    r_sc = X[0:3]
    r_earth, _ = earth_state_func(float(t))
    r_sun, _ = sun_state_func(float(t))

    A = srp_thirdbody_variational_eq_for6state(
        r_sc=r_sc,
        r_earth=r_earth,
        r_sun=r_sun,
        Cr=float(Cr),
        area=scConst.area,
        mass=scConst.mass,
        mu_earth=pConst.mu_earth,
        mu_i=pConst.mu_sun,
        solar_flux_1au=scConst.solar_flux_1au,
        c=scConst.c,
        AU_m=scConst.AU_m,
    )

    dPhi = (A @ Phi).reshape(-1)
    return np.hstack((dX, dPhi))

