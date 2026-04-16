import numpy as np
import pandas as pd
import sys
import scipy.io
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("../../")

from src.Functions.filters import UnscentedKalmanFilter, LinearizedKalmanFilter, ExtendedKalmanFilter
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_stm_deriv, mu_sun_srp_state_deriv
from src.Functions.jacobians import cannonball_SRP, srp_thirdbody_variational_eq
from src.Functions.Ephem import ephem
from src.Functions.calcB_plane import calc_bplane
from src.Functions.SOIcheck import SOIcheck
from src.helpers.plotting.plot_bplane import make_bplane_plot

def build_stations():
    theta0_deg = 0
    w_earth_rad_per_s = 7.29211585275553e-5
    radius_earth_km = 6378.1363

    return [
        Stations(
            "DSS 34",
            lat_deg=-35.398333,
            lon_deg=148.981944,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + .691750,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "DSS 65",
            lat_deg=40.427222,
            lon_deg=355.749444,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + .834539,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "DSS 13",
            lat_deg=35.247164,
            lon_deg=243.205000,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + 1.07114904,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
    ]


# Initial UKF state/covariance scaffold for 7-state [r, v, Cr].
x0_ukf = np.array([ -274096770.76544, -92859266.4499061, -40199493.6677441, 32.6704564599943,   -8.93838913761049,  -3.87881914050316, 1], dtype=float)
P0_ukf = np.diag(
    [
        100.0**2,     # x [km]
        100.0**2,     # y [km]
        100.0**2,     # z [km]
        .1**2,    # vx [km/s]
        .1**2,    # vy [km/s]
        .1**2,    # vz [km/s]
        .1**2,    # Cr [-]
    ],
)

