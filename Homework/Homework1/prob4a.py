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

# Collect all measurements in one list
all_meas = []

for k in range(len(t)):
    for st in stations:
        m = st.measure(r[k], v[k], float(t[k]))
        if m is not None:
            all_meas.append(m)

# First/last measurement times

t_first = min(m["t"] for m in all_meas)
t_last  = max(m["t"] for m in all_meas)
print("First measurement time (s):", t_first)
print("Last measurement time (s):", t_last)






# Plot by station: Range
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue
    tt  = np.array([m["t"] for m in st_meas], dtype=float)
    rho = np.array([m["rho_km"] for m in st_meas], dtype=float)
    plt.scatter(tt, rho, s=6, alpha=0.8, label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Range ρ (km)")
plt.title("Range Measurements)")
plt.grid(True)
plt.legend()


# Plot by station: Range-rate
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue
    tt   = np.array([m["t"] for m in st_meas], dtype=float)
    rhod = np.array([m["rho_dot_km_s"] for m in st_meas], dtype=float)
    plt.scatter(tt, rhod, s=6, alpha=0.8, label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Range-rate ρ̇ (km/s)")
plt.title("Range-rate measurements")
plt.grid(True)
plt.legend()


# Plot by station: Elevation
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue
    tt   = np.array([m["t"] for m in st_meas], dtype=float)
    elev = np.array([np.rad2deg(m["elev_rad"]) for m in st_meas], dtype=float)
    plt.scatter(tt, elev, s=6, alpha=0.8, label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Elevation (deg)")
plt.title("Elevation Angles at Measurement Times")
plt.grid(True)
plt.legend()

plt.show()





################### D ############################

def add_gaussian_noise(y: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return y + rng.normal(loc=0.0, scale=sigma, size=y.shape)


sigma_km_s = 5e-7  # 0.5 mm/s in km/s

# Original vs noisy
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue

    tt = np.array([m["t"] for m in st_meas], dtype=float)
    rhod = np.array([m["rho_dot_km_s"] for m in st_meas], dtype=float)
    rhod_noisy = add_gaussian_noise(rhod, sigma_km_s, seed=123)

    plt.scatter(tt, rhod,       s=6, alpha=0.8, label=f"{st.name} original")
    plt.scatter(tt, rhod_noisy, s=6, alpha=0.6, label=f"{st.name} noisy")

plt.xlabel("t (s)")
plt.ylabel("Range-rate ρ̇ (km/s)")
plt.title("Range-rate: Original vs Noisy (σ = 0.5 mm/s)")
plt.grid(True)
plt.legend()


# Residual plot
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue

    tt = np.array([m["t"] for m in st_meas], dtype=float)
    rhod = np.array([m["rho_dot_km_s"] for m in st_meas], dtype=float)
    rhod_noisy = add_gaussian_noise(rhod, sigma_km_s, seed=123)
    residual = rhod_noisy - rhod

    plt.scatter(tt, residual, s=6, alpha=0.8, label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Noisy − original (km/s)")
plt.title("Range-Rate Noise Residuals")
plt.grid(True)
plt.legend()

plt.show()
