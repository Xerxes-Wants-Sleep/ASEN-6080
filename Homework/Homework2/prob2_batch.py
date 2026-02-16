import numpy as np
import sys
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append("../../")

from src.Functions.stations import Stations
from src.Functions.batch import batch_estimate_x0, postprocess_batch_for_plots




meas_path = "meas_data/prob2_hw2_measurements_noisy.csv"  
df = pd.read_csv(meas_path)

# Convert to list-of-dicts like your batch expects
all_meas = df.to_dict(orient="records")

# -----------------------------
# 2) Stations (same as HW1)
# -----------------------------
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944, theta0_deg=0.0, radius_earth=6378.0, w_earth_rad_per_s=7.2921158553e-5),
    Stations("Station 2", lat_deg=40.427222, lon_deg=355.749444, theta0_deg=0.0, radius_earth=6378.0, w_earth_rad_per_s=7.2921158553e-5),
    Stations("Station 3", lat_deg=35.247164, lon_deg=243.205000, theta0_deg=0.0, radius_earth=6378.0, w_earth_rad_per_s=7.2921158553e-5),
]

# 3) Define measurement noise R
sigma_rho_km = 1.0e-3          # 1 m = 1e-3 km
sigma_rhod_km_s = 1.0e-6       # 1 mm/s = 1e-6 km/s

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_PLOTS_DIR = SCRIPT_DIR / "Plots" / "Batch 2B"
BATCH_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

# -----------------------------
# 4) Define a priori covariance P0
#    σr = 1 km, σv = 1 m/s = 1e-3 km/s
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
# 5) Initial guess x0_bar
#    You said you have dx = [0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3]
#
#    IMPORTANT: x0_bar must be an actual initial state guess.
#    If you have a nominal truth/reference x0_true, you'd do:
#      x0_bar = x0_true + dx
#
#    If you *don't* have x0_true handy here yet, you must load it from
#    the truth CSV (first row of prob2a_traj.csv), or however HW specifies.
# -----------------------------



dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
# dx = 100 * np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)





# Example: load truth IC from the truth trajectory CSV first row
truth_path = "../Homework1/HW2_j3_on_truth.csv"
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
# 6) Dynamics parameters (fill with your assignment values)
# -----------------------------
# NOTE: put the actual numbers your HW uses.
# If your HW1 had these in a JSON or constants section, reuse them.
mu = 398600.4415   # km^3/s^2 (example Earth)
J2 = 0.0010826269
J3 = -2.5324e-6
# -----------------------------
# 7) Run batch
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

print("\n=== Batch finished ===")
print("num_iters:", info["num_iters"])
print("x0_hat:", x0_hat)
print("diag(P0_hat):", np.diag(P0_hat))

# quick residual sanity
postfit = info["postfit_resids"]
print("postfit residuals shape:", postfit.shape)
print("postfit mean [rho, rhod]:", postfit.mean(axis=0))
print("postfit std  [rho, rhod]:", postfit.std(axis=0))

# -----------------------------
# 8) Postprocess (no plotting yet)
#    Pass truth if you want error vs 2sigma later
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
    x0_star_hist=info["x0_star_hist"],   # <-- ADD THIS
    first_pass_gap_s=6*3600.0,           # optional; tune if you want
)


# print("\n=== Postprocess finished ===")
# print("xhat_meas shape:", plot_data["xhat_meas"].shape)
# print("P_meas shape:", plot_data["P_meas"].shape)
# print("two_sigma shape:", plot_data["two_sigma_meas"].shape)
# if plot_data["state_error_meas"] is None:
#     print("state_error_meas: None (truth not provided)")
# else:
#     print("state_error_meas shape:", plot_data["state_error_meas"].shape)

# print("postfit_resids_meas shape:", plot_data["postfit_resids_meas"].shape)




### Plot pretty please

def save_fig(fig, filename, dpi=300):
    outpath = BATCH_PLOTS_DIR / filename
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

SHOW_PLOTS = False  # set True if you want windows to pop up


# -----------------------------
# (1) State error vs time with ±3σ bounds
# -----------------------------
t = plot_data["t_meas"]
err = plot_data["state_error_meas"]
two_sigma = plot_data["two_sigma_meas"]

if err is None:
    raise ValueError("plot_data['state_error_meas'] is None. Make sure you passed truth_times and truth_states_6.")

three_sigma = (3.0/2.0) * two_sigma  # stored 2σ -> convert to 3σ
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
fig.suptitle("Batch State Error vs Time with ±3σ Bounds", y=0.98)
plt.tight_layout()

save_fig(fig, "batch_state_error_all.png")


# -----------------------------
# (2) Postfit residuals vs time (±3σ measurement noise)
# -----------------------------
post = plot_data["postfit_resids_meas"]  # (m,2): [rho_km, rho_dot_km_s]

# Use same sigmas you used in R
sigma_rho_km = 1.0e-3     # 1 m
sigma_rhod_km_s = 1.0e-6  # 1 mm/s

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

fig.suptitle("Postfit Residuals vs Time (with ±3σ noise bounds)", y=0.98)
plt.tight_layout()

save_fig(fig, "batch_postfit_residuals.png")


# -----------------------------
# (3) RMS summary table (prints only; no figure)
# -----------------------------
rms = plot_data["rms_final"]
rows = []

rows.append({
    "Metric": "Postfit RMS: rho (km)",
    "All meas": rms["postfit_all"][0],
    "Ignore first pass": rms["postfit_ignore_first"][0],
})
rows.append({
    "Metric": "Postfit RMS: rho_dot (km/s)",
    "All meas": rms["postfit_all"][1],
    "Ignore first pass": rms["postfit_ignore_first"][1],
})

if rms["state_comp_all"] is not None:
    labels_state = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]
    for i, lab in enumerate(labels_state):
        rows.append({
            "Metric": f"State RMS: {lab}",
            "All meas": rms["state_comp_all"][i],
            "Ignore first pass": rms["state_comp_ignore_first"][i],
        })

    rows.append({
        "Metric": "3D RMS: ||r|| (km)",
        "All meas": rms["pos3_all"],
        "Ignore first pass": rms["pos3_ignore_first"],
    })
    rows.append({
        "Metric": "3D RMS: ||v|| (km/s)",
        "All meas": rms["vel3_all"],
        "Ignore first pass": rms["vel3_ignore_first"],
    })

df_rms = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:.3e}")
print("\n=== RMS Summary (Final Batch) ===")
print(df_rms.to_string(index=False))


# -----------------------------
# (4) RMS vs iteration plots
# -----------------------------
rms_iter = plot_data["rms_by_iter"]
if rms_iter is None:
    print("No rms_by_iter returned. Did you pass x0_star_hist=info['x0_star_hist'] to postprocess?")
else:
    K = rms_iter["postfit_all"].shape[0]
    it = np.arange(K)  # iter index (0 = initial guess)

    # (4a) Postfit RMS vs iteration
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(it, rms_iter["postfit_all"][:, 0], marker="o", label="rho RMS (all)")
    ax.plot(it, rms_iter["postfit_all"][:, 1], marker="o", label="rho_dot RMS (all)")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.set_xlabel("Iteration index")
    ax.set_ylabel("RMS")
    ax.set_title("Postfit Residual RMS vs Iteration (Batch)")
    ax.legend()
    plt.tight_layout()

    save_fig(fig, "batch_postfit_rms_vs_iteration.png")

    # (4b) State RMS vs iteration (only if truth exists)
    if rms_iter["state_comp_all"] is not None:
        # 3D RMS vs iteration
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(it, rms_iter["pos3_all"], marker="o", label="||r|| RMS (all)")
        ax.plot(it, rms_iter["vel3_all"], marker="o", label="||v|| RMS (all)")
        ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.set_xlabel("Iteration index")
        ax.set_ylabel("RMS")
        ax.set_title("3D State RMS vs Iteration (Batch)")
        ax.legend()
        plt.tight_layout()

        save_fig(fig, "batch_state_3d_rms_vs_iteration.png")

        # Component RMS vs iteration
        labels_state = ["x", "y", "z", "vx", "vy", "vz"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(6):
            ax.plot(it, rms_iter["state_comp_all"][:, i], marker="o", label=labels_state[i])
        ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.set_xlabel("Iteration index")
        ax.set_ylabel("RMS")
        ax.set_title("State Component RMS vs Iteration (Batch)")
        ax.legend(ncol=3)
        plt.tight_layout()

        save_fig(fig, "batch_state_component_rms_vs_iteration.png")


