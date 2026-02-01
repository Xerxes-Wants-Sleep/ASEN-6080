import numpy as np
import sys
import pandas as pd

sys.path.append("../../")

from src.Functions.stations import Stations

# file paths stuff
truth_path = "../../Homework/Homework1/HW2_j3_on_truth.csv"
out_path_clean = "../../Homework/Homework2/meas_data/prob2_hw2_measurements_clean.csv"
out_path_noisy = "../../Homework/Homework2/meas_data/prob2_hw2_measurements_noisy.csv"

truth = pd.read_csv(truth_path)
t = truth["t_s"].to_numpy(float)
r = truth[["x_km", "y_km", "z_km"]].to_numpy(float)
v = truth[["vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)

# Stations
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]

# Collect measurements
all_meas = []
for k in range(len(t)):
    for st in stations:
        m = st.measure(r[k], v[k], float(t[k]))
        if m is not None:
            all_meas.append(m)


t_first = all_meas[0]["t"]
t_last  = all_meas[-1]["t"]

# Convert to DataFrame
df = pd.DataFrame(all_meas)  # station, t, rho_km, rho_dot_km_s, elev_rad

# Save clean measurements in case
df.to_csv(out_path_clean, index=False)
print("Saved clean:", out_path_clean)

#  Add Gaussian noise for HW2
sigma_rho_km = 1e-3
sigma_rhod_km_s = 1e-6

seed = 123  # for repeatability
rng = np.random.default_rng(seed)

df_noisy = df.copy()
df_noisy["rho_km"] = df_noisy["rho_km"] + rng.normal(0.0, sigma_rho_km, size=len(df_noisy))
df_noisy["rho_dot_km_s"] = df_noisy["rho_dot_km_s"] + rng.normal(0.0, sigma_rhod_km_s, size=len(df_noisy))

df_noisy.to_csv(out_path_noisy, index=False)
print("Saved noisy:", out_path_noisy)

