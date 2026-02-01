import numpy as np
import sys
import json
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")

from src.Functions.stations import Stations

truth = pd.read_csv("prob2a_traj.csv")
t = truth["t_s"].to_numpy(float)
r = truth[["x_km", "y_km", "z_km"]].to_numpy(float)
v = truth[["vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)

# Stations (calling class)
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]




