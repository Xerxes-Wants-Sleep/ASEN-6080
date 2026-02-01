import numpy as np
import sys
import json
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")

from src.Functions.stations import Stations





def RUdoppler( station: Stations, r_sc_eci: np.ndarray, v_sc_eci: np.ndarray, t: float, f_tr_ref_hz: float = 8.44e9, c_km_s: float = 2.99792458e5) -> dict:
    elev = station.elevation(t, r_sc_eci)

    if elev < station.elevation_mask_rad:
        return {"station": station.name, "t": t, "RU": np.nan, "f_shift_hz": np.nan, "elev_rad": elev}

    m = station.measure(r_sc_eci, v_sc_eci, t)
    rho_km = float(m["rho_km"])
    rhodot_km_s = float(m["rho_dot_km_s"])

    f_shift = -2.0 * rhodot_km_s * f_tr_ref_hz / c_km_s
    RU = (221.0 / 749.0) * (rho_km / c_km_s) * f_tr_ref_hz

    return {"station": station.name, "t": t, "RU": RU, "f_shift_hz": f_shift, "elev_rad": elev}



truth = pd.read_csv("prob2a_traj.csv")
t = truth["t_s"].to_numpy(float)
r = truth[["x_km", "y_km", "z_km"]].to_numpy(float)
v = truth[["vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)

stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]

all_dsn = []

for k in range(len(t)):
    for st in stations:
        y = RUdoppler(st, r[k], v[k], float(t[k]))  # returns dict
        all_dsn.append(y)


plt.figure()
for st in stations:
    st_dsn = [d for d in all_dsn if d["station"] == st.name and np.isfinite(d["RU"])]
    if not st_dsn:
        continue
    plt.plot([d["t"] for d in st_dsn],
             [d["RU"] for d in st_dsn],
             marker=".", linestyle="None", label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Range Units (RU)")
plt.title("2-way Range in Range Units (RU)")
plt.grid(True)
plt.legend()

plt.figure()
for st in stations:
    st_dsn = [d for d in all_dsn if d["station"] == st.name and np.isfinite(d["f_shift_hz"])]
    if not st_dsn:
        continue
    plt.plot([d["t"] for d in st_dsn],
             [d["f_shift_hz"] for d in st_dsn],
             marker=".", linestyle="None", label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Doppler shift (Hz)")
plt.title("2-way Doppler shift (Hz)")
plt.grid(True)
plt.legend()
plt.show()