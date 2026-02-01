import numpy as np
import sys
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append("../../")

from src.Functions.stations import Stations
from src.Functions.batch import batch_estimate_x0, postprocess_batch_for_plots

meas_path = "meas_data/hw2_measurements_noisy.csv"
df = pd.read_csv(meas_path)

# Convert to list-of-dicts like your batch expects
all_meas_full = df.to_dict(orient="records")

# IMPORTANT: ensure time-sorted
all_meas_full = sorted(all_meas_full, key=lambda m: float(m["t"]))

# -----------------------------
# (F) Only process FIRST HALF of the measurements (by time midpoint)
# -----------------------------
t0_meas = float(all_meas_full[0]["t"])
tf_meas = float(all_meas_full[-1]["t"])
tmid = 0.5 * (t0_meas + tf_meas)

all_meas = [m for m in all_meas_full if float(m["t"]) <= tmid]

print(f"\n=== Part (F): using only first half of arc ===")
print(f"t0 = {t0_meas:.3f} s, tf = {tf_meas:.3f} s, tmid = {tmid:.3f} s")
print(f"Using {len(all_meas)} / {len(all_meas_full)} measurements")

# -----------------------------
# 2) Stations (same as HW1)
# -----------------------------
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]

# 3) Define measurement noise R (ORIGINAL noise)
sigma_rho_km = 1.0e-3          # 1 m = 1e-3 km
sigma_rhod_km_s = 1.0e-6       # 1 mm/s = 1e-6 km/s

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_PLOTS_DIR = SCRIPT_DIR / "Plots" / "Batch for F"
BATCH_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

# -----------------------------
# 4) Define a priori covariance P0 (ORIGINAL P0)
#    σr = 1 km, σv = 1 m/s = 1e-3 km/s
# -----------------------------
sigma_r0_km = 1.0
sigma_v0_km_s = 1.0e-3

P0 = np.diag([
    sigma_r0_km**2, sigma_r0_km**2, sigma_r0_km**2,
    sigma_v0_km_s**2, sigma_v0_km_s**2, sigma_v0_km_s**2
])

# -----------------------------
# 5) Initial guess x0_bar (ORIGINAL dx0)
# -----------------------------
dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)

truth_path = "../Homework1/prob2a_traj.csv"
truth_df = pd.read_csv(truth_path)
x0_true = np.array([
    truth_df.loc[0, "x_km"],
    truth_df.loc[0, "y_km"],
    truth_df.loc[0, "z_km"],
    truth_df.loc[0, "vx_km_s"],
    truth_df.loc[0, "vy_km_s"],
    truth_df.loc[0, "vz_km_s"],
], dtype=float)

x0_bar = x0_true + dx

# -----------------------------
# 6) Dynamics parameters
# -----------------------------
mu = 398600.4418
J2 = 1.08262668e-3
J3 = -2.532153e-6

# -----------------------------
# 7) Run batch (FIRST HALF ONLY)
# -----------------------------
x0_hat, P0_hat, info = batch_estimate_x0(
    all_meas=all_meas,
    stations=stations,
    x0_bar=x0_bar,
    P0=P0,
    R=R,
    mu=mu,
    J2=J2,
    J3=J3,
    max_iter=10,
    tol=1e-10,
    reltol=1e-10,
    abstol=1e-10,
)

print("\n=== Batch finished (F) ===")
print("num_iters:", info["num_iters"])
print("x0_hat:", x0_hat)
print("diag(P0_hat):", np.diag(P0_hat))

postfit = info["postfit_resids"]
print("postfit residuals shape:", postfit.shape)
print("postfit mean [rho, rhod]:", postfit.mean(axis=0))
print("postfit std  [rho, rhod]:", postfit.std(axis=0))

# -----------------------------
# 8) Postprocess (FIRST HALF ONLY)
# -----------------------------
truth_times = truth_df["t_s"].to_numpy(float)
truth_states_6 = truth_df[["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]].to_numpy(float)

plot_data = postprocess_batch_for_plots(
    all_meas=all_meas,
    stations=stations,
    x0_hat=x0_hat,
    P0_hat=P0_hat,
    mu=mu,
    J2=J2,
    J3=J3,
    truth_times=truth_times,
    truth_states_6=truth_states_6,
    reltol=1e-10,
    abstol=1e-10,
    x0_star_hist=info["x0_star_hist"],
    first_pass_gap_s=6*3600.0,
)

def save_fig(fig, filename, dpi=300):
    outpath = BATCH_PLOTS_DIR / filename
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

# -----------------------------
# (1) State error vs time with ±3σ bounds
# -----------------------------
t = plot_data["t_meas"]
err = plot_data["state_error_meas"]
two_sigma = plot_data["two_sigma_meas"]

if err is None:
    raise ValueError("plot_data['state_error_meas'] is None. Make sure you passed truth_times and truth_states_6.")

three_sigma = (3.0/2.0) * two_sigma
labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]

fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()

for i in range(6):
    ax = axs[i]
    ax.scatter(t, err[:, i], s=6, marker="o", label="state error")
    ax.plot(t, +three_sigma[:, i], linestyle="--", linewidth=1.5, color="k", label="+3σ")
    ax.plot(t, -three_sigma[:, i], linestyle="--", linewidth=1.5, color="k", label="-3σ")
    ax.set_ylabel(labels[i])
    ax.grid(True)

axs[-2].set_xlabel("t (s)")
axs[-1].set_xlabel("t (s)")
handles, leglabels = axs[0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc="upper right")
fig.suptitle("Batch State Error vs Time with ±3σ Bounds (First Half Only)", y=0.98)
plt.tight_layout()

save_fig(fig, "batch_state_error_first_half.png")

# -----------------------------
# (2) Postfit residuals vs time (±3σ measurement noise)
# -----------------------------
post = plot_data["postfit_resids_meas"]

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

fig.suptitle("Postfit Residuals vs Time (First Half Only)", y=0.98)
plt.tight_layout()

save_fig(fig, "batch_postfit_residuals_first_half.png")

# -----------------------------
# (3) RMS summary table (prints only; no figure)
# -----------------------------
rms = plot_data["rms_final"]
rows = []

rows.append({"Metric": "Postfit RMS: rho (km)",     "All meas": rms["postfit_all"][0], "Ignore first pass": rms["postfit_ignore_first"][0]})
rows.append({"Metric": "Postfit RMS: rho_dot (km/s)","All meas": rms["postfit_all"][1], "Ignore first pass": rms["postfit_ignore_first"][1]})

if rms["state_comp_all"] is not None:
    labels_state = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]
    for i, lab in enumerate(labels_state):
        rows.append({"Metric": f"State RMS: {lab}", "All meas": rms["state_comp_all"][i], "Ignore first pass": rms["state_comp_ignore_first"][i]})
    rows.append({"Metric": "3D RMS: ||r|| (km)", "All meas": rms["pos3_all"], "Ignore first pass": rms["pos3_ignore_first"]})
    rows.append({"Metric": "3D RMS: ||v|| (km/s)", "All meas": rms["vel3_all"], "Ignore first pass": rms["vel3_ignore_first"]})

df_rms = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:.3e}")
print("\n=== RMS Summary (Final Batch, First Half Only) ===")
print(df_rms.to_string(index=False))

# -----------------------------
# (4) RMS vs iteration plots
# -----------------------------
rms_iter = plot_data["rms_by_iter"]
if rms_iter is None:
    print("No rms_by_iter returned. Did you pass x0_star_hist=info['x0_star_hist'] to postprocess?")
else:
    K = rms_iter["postfit_all"].shape[0]
    it = np.arange(K)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(it, rms_iter["postfit_all"][:, 0], marker="o", label="rho RMS (all)")
    ax.plot(it, rms_iter["postfit_all"][:, 1], marker="o", label="rho_dot RMS (all)")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.set_xlabel("Iteration index")
    ax.set_ylabel("RMS")
    ax.set_title("Postfit Residual RMS vs Iteration (Batch, First Half Only)")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "batch_postfit_rms_vs_iteration_first_half.png")

    if rms_iter["state_comp_all"] is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(it, rms_iter["pos3_all"], marker="o", label="||r|| RMS (all)")
        ax.plot(it, rms_iter["vel3_all"], marker="o", label="||v|| RMS (all)")
        ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.set_xlabel("Iteration index")
        ax.set_ylabel("RMS")
        ax.set_title("3D State RMS vs Iteration (Batch, First Half Only)")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "batch_state_3d_rms_vs_iteration_first_half.png")

        labels_state = ["x", "y", "z", "vx", "vy", "vz"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(6):
            ax.plot(it, rms_iter["state_comp_all"][:, i], marker="o", label=labels_state[i])
        ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.set_xlabel("Iteration index")
        ax.set_ylabel("RMS")
        ax.set_title("State Component RMS vs Iteration (Batch, First Half Only)")
        ax.legend(ncol=3)
        plt.tight_layout()
        save_fig(fig, "batch_state_component_rms_vs_iteration_first_half.png")
