import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


sys.path.append("../../")

from src.Functions.stations import Stations
from src.Functions.filters import LinearizedKalmanFilter 

# -----------------------------
# 1) Read measurements (noisy)  (SAME AS BATCH)
# -----------------------------
meas_path = "meas_data/hw2_measurements_noisy.csv"
df = pd.read_csv(meas_path)

all_meas = df.to_dict(orient="records")
all_meas = sorted(all_meas, key=lambda m: float(m["t"]))

t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)

# -----------------------------
# 2) Stations (same as HW1 / batch)
# -----------------------------
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]


# -----------------------------
# 3) Measurement noise R
# -----------------------------
sigma_rho_km = 1.0e-3          # 1 m = 1e-3 km
sigma_rhod_km_s = 1.0e-6       # 1 mm/s = 1e-6 km/s

R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])



PLOTS_DIR = Path("Plots") / "LKF for part B"     # <- subfolder for LKF figures
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 4) A priori covariance P0
# -----------------------------
sigma_r0_km = 1.0
sigma_v0_km_s = 1.0e-3

# sigma_r0_km = 1.0e3
# sigma_v0_km_s = 1.0


P0 = np.diag([
    sigma_r0_km**2, sigma_r0_km**2, sigma_r0_km**2,
    sigma_v0_km_s**2, sigma_v0_km_s**2, sigma_v0_km_s**2
])


# -----------------------------
# 5) Process noise Q (DISCRETE)  (pass zeros for now)
# -----------------------------
Q = np.zeros((6, 6), dtype=float)


# -----------------------------
# 6) Truth + initial guess X0_star = X0_true + dx  (SAME AS BATCH)
# -----------------------------
dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)

# dx = 100 * np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)




truth_path = "../Homework1/prob2a_traj.csv"
truth_df = pd.read_csv(truth_path)

# initial truth state (first row)
x0_true = np.array([
    truth_df.loc[0, "x_km"],
    truth_df.loc[0, "y_km"],
    truth_df.loc[0, "z_km"],
    truth_df.loc[0, "vx_km_s"],
    truth_df.loc[0, "vy_km_s"],
    truth_df.loc[0, "vz_km_s"],
], dtype=float)

X0_star = x0_true + dx

truth_times = truth_df["t_s"].to_numpy(float)
truth_states = truth_df[["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]].to_numpy(float)

Xtrue_meas = np.column_stack([np.interp(t_meas, truth_times, truth_states[:,k]) for k in range(6)])

# -----------------------------
# 7) Dynamics parameters
# -----------------------------
mu = 398600.4415
J2 = 0.0010826269
J3 = -2.5324e-6



# -----------------------------
# 8) Run LKF
# -----------------------------
lkf = LinearizedKalmanFilter(
    X0_star=X0_star,
    P0=P0,
    R=R,
    Q=Q,
    mu=mu,
    J2=J2,
    J3=J3,
    reltol=1e-10,
    abstol=1e-10,
    method="DOP853",
    j2=True,
    j3=False,
    first_pass_gap_s=6*3600.0,
)

out = lkf.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)

print("\n=== LKF finished ===")
print("Xhat_meas shape:", out["Xhat_meas"].shape)
print("postfit_resids_meas shape:", out["postfit_resids_meas"].shape)









# -----------------------------
# Plot output settings
# -----------------------------


def save_fig(fig, filename, dpi=300):
    outpath = PLOTS_DIR / filename
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath.resolve()}")



# -----------------------------
# 9) Plot: state error vs time
# -----------------------------
t = out["t_meas"]
err = out["state_error_meas"]          # (m,6)
labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]

fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()

for i in range(6):
    ax = axs[i]
    ax.scatter(t, err[:, i], s=6, marker="o", label="state error")
    ax.set_ylabel(labels[i])
    ax.grid(True)

# overlay ±3σ if available
if "two_sigma_meas" in out and out["two_sigma_meas"] is not None:
    three_sigma = (3.0 / 2.0) * out["two_sigma_meas"]
    for i in range(6):
        axs[i].plot(t, +three_sigma[:, i], linestyle="--", linewidth=1.5, color="k", label="+3σ")
        axs[i].plot(t, -three_sigma[:, i], linestyle="--", linewidth=1.5, color="k", label="-3σ")

axs[-2].set_xlabel("t (s)")
axs[-1].set_xlabel("t (s)")
handles, leglabels = axs[0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc="upper right")
fig.suptitle("LKF State Error vs Time (±3σ if available)", y=0.98)
plt.tight_layout()

save_fig(fig, "lkf_state_error_all.png")


# -----------------------------
# 10) Plot: postfit residuals with ±3σ measurement noise bounds
# -----------------------------
post = out["postfit_resids_meas"]  # (m,2)

fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axs[0].scatter(t, post[:, 0], s=6, marker="o")
axs[0].axhline(0.0, linewidth=1)
axs[0].axhline(+3*sigma_rho_km, linestyle="--", linewidth=1.5, color="k")
axs[0].axhline(-3*sigma_rho_km, linestyle="--", linewidth=1.5, color="k")
axs[0].set_ylabel("Postfit ρ residual (km)")
axs[0].grid(True)

axs[1].scatter(t, post[:, 1], s=6, marker="o")
axs[1].axhline(0.0, linewidth=1)
axs[1].axhline(+3*sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k")
axs[1].axhline(-3*sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k")
axs[1].set_ylabel("Postfit ρ̇ residual (km/s)")
axs[1].set_xlabel("t (s)")
axs[1].grid(True)

fig.suptitle("LKF Postfit Residuals vs Time (±3σ noise bounds)", y=0.98)
plt.tight_layout()

save_fig(fig, "lkf_postfit_residuals.png")


# -----------------------------
# 11) Print RMS summary table
# -----------------------------
rms = out["rms_final"]
rows = [
    {"Metric": "Postfit RMS: rho (km)",      "All meas": rms["postfit_all"][0],          "Ignore first pass": rms["postfit_ignore_first"][0]},
    {"Metric": "Postfit RMS: rho_dot (km/s)","All meas": rms["postfit_all"][1],          "Ignore first pass": rms["postfit_ignore_first"][1]},
]

if rms["state_comp_all"] is not None:
    for i, lab in enumerate(labels):
        rows.append({"Metric": f"State RMS: {lab}", "All meas": rms["state_comp_all"][i], "Ignore first pass": rms["state_comp_ignore_first"][i]})
    rows.append({"Metric": "3D RMS: ||r|| (km)",   "All meas": rms["pos3_all"],         "Ignore first pass": rms["pos3_ignore_first"]})
    rows.append({"Metric": "3D RMS: ||v|| (km/s)", "All meas": rms["vel3_all"],         "Ignore first pass": rms["vel3_ignore_first"]})

df_rms = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:.3e}")
print("\n=== RMS Summary (LKF) ===")
print(df_rms.to_string(index=False))


# -----------------------------
# 12) Plot: state error vs time (Zoom: t >= 4000 s)
# -----------------------------
t0_zoom = 20000.0
mask = t >= t0_zoom

t_zoom = t[mask]
err_zoom = err[mask, :]

fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()

for i in range(6):
    ax = axs[i]
    ax.scatter(t_zoom, err_zoom[:, i], s=6, marker="o", label="state error")
    ax.set_ylabel(labels[i])
    ax.grid(True)

# overlay ±3σ if available (same mask)
if "two_sigma_meas" in out and out["two_sigma_meas"] is not None:
    three_sigma = (3.0 / 2.0) * out["two_sigma_meas"]
    three_sigma_zoom = three_sigma[mask, :]
    for i in range(6):
        axs[i].plot(t_zoom, +three_sigma_zoom[:, i], linestyle="--", linewidth=1.5, color="k", label="+3σ")
        axs[i].plot(t_zoom, -three_sigma_zoom[:, i], linestyle="--", linewidth=1.5, color="k", label="-3σ")

axs[-2].set_xlabel("t (s)")
axs[-1].set_xlabel("t (s)")
handles, leglabels = axs[0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc="upper right")
fig.suptitle(f"LKF State Error vs Time (Zoom: t ≥ {t0_zoom:.0f} s)", y=0.98)
plt.tight_layout()

save_fig(fig, f"lkf_state_error_zoom_t_ge_{int(t0_zoom)}.png")
