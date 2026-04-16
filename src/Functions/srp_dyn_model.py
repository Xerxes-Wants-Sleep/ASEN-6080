import numpy as np
from .jacobians import cannonball_SRP, srp_thirdbody_variational_eq

def srp_dyn(mu: float,
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

    r = r_sc - r_earth

    x, y, z = r
    r2 = x*x + y*y + z*z
    rmag = np.sqrt(r2)

    # Central term
    a_mu = -mu * r / (rmag**3)

    #Third body acceeration
    r_i_sc = r_sun[0:3] - r_sc[0:3]
    r_i_earth = r_sun[0:3] - r_earth
    a_i = mu_i * (r_i_sc / np.linalg.norm(r_i_sc)**3 - r_i_earth / np.linalg.norm(r_i_earth)**3)


    # SRP Acceleration
    a_srp, _, _ = cannonball_SRP(
        r_sc = r_sc[0:3],
        r_sun = r_sun[0:3],
        Cr = Cr,
        area = area,
        mass = mass,
        solar_flux_1au = solar_flux_1au,  
        c = c,
        AU_m = AU_m,       
    )
    a = a_mu + a_i + a_srp
    return a

def mu_sun_srp_state_deriv(
    t: float,
    X: np.ndarray,
    pConst,
    scConst,
    earth_state_func,
    sun_state_func,
):
    """
     7-state
        X = [x, y, z, xdot, ydot, zdot, Cr]
    """

    X = np.asarray(X, dtype=float).reshape(7)

    r_sc = X[0:3]
    v_sc = X[3:6]
    Cr = X[6]

    r_earth, _ = earth_state_func(t)
    r_sun, _ = sun_state_func(t)

    a = srp_dyn(
        mu=pConst.mu_earth,
        mu_i=pConst.mu_sun,
        r_sc=r_sc,
        r_earth=r_earth,
        r_sun=r_sun,
        Cr=Cr,
        area=scConst.area,
        mass=scConst.mass,
        solar_flux_1au=scConst.solar_flux_1au,
        c=scConst.c,
        AU_m=scConst.AU_m,
    )

    dX = np.zeros(7, dtype=float)
    dX[0:3] = v_sc
    dX[3:6] = a
    dX[6] = 0.0

    return dX


def mu_sun_srp_stm_deriv(
    t: float,
    XPhi: np.ndarray,
    pConst,
    scConst,
    earth_state_func,
    sun_state_func,
):
    """
    Combined state + STM derivative for ODE integration.

    Parameters
    ----------
    t : float
        Current integration time [s].
    XPhi : ndarray, shape (56,)
        Stacked vector ``[X; Phi_flat]`` where:
          - ``X`` is the 7-state ``[x, y, z, xdot, ydot, zdot, Cr]``
          - ``Phi_flat`` is the flattened 7x7 STM.
    pConst : object
        Constants container with attributes:
          - ``mu_earth`` : Earth gravitational parameter
          - ``mu_sun``   : Sun gravitational parameter
        Units must be consistent with the state/ephemeris units.
    scConst : object
        Spacecraft/SRP constants container with attributes:
          - ``area`` : effective SRP area
          - ``mass`` : spacecraft mass
          - ``solar_flux_1au`` : solar flux at 1 AU
          - ``c`` : speed of light
          - ``AU_m`` : astronomical unit in meters
    earth_state_func : callable
        Function handle ``earth_state_func(t) -> (r_earth, v_earth)``.
        Must return Earth position/velocity in the same frame and units as ``X``.
    sun_state_func : callable
        Function handle ``sun_state_func(t) -> (r_sun, v_sun)``.
        Must return Sun position/velocity in the same frame and units as ``X``.

    Returns
    -------
    ndarray, shape (56,)
        Stacked derivative ``[dX/dt; d(Phi_flat)/dt]``.
    """
    XPhi = np.asarray(XPhi, dtype=float).reshape(-1)

    n = 7
    X = XPhi[:n]
    Phi = XPhi[n:].reshape(n, n)

    # Nonlinear state derivative
    dX = mu_sun_srp_state_deriv(
        t=t,
        X=X,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    # Current ephemeris states
    r_sc = X[0:3]
    Cr = X[6]
    r_earth, _ = earth_state_func(t)
    r_sun, _ = sun_state_func(t)

    # Continuous-time Jacobian
    A = srp_thirdbody_variational_eq(
        r_sc=r_sc,
        r_earth=r_earth,
        r_sun=r_sun,
        Cr=Cr,
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
