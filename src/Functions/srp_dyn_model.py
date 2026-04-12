import numpy as np
from .jacobians import (cannonball_SRP)

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