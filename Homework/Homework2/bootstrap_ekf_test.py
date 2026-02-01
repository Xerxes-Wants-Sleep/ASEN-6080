
############### DOES NOT WORK ###################






import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../../")

from src.Functions.stations import Stations
from src.Functions.bootstrap_ekf import warmstart_ekf_with_lkf


# -----------------------------
# 1) Read measurements (noisy)
# -----------------------------
meas_path = "meas_data/hw2_measurements_noisy.csv"
df = pd.read_csv(meas_path)

all_meas = df.to_dict(orient="records")
all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)

# -----------------------------
# 2) Stations
# -----------------------------
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]

# -----------------------------
# 3) Measurement noise R
# -----------------------------
sigma_rho_km = 1.0e-3
sigma_rhod_km_s = 1.0e-6
R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

# -----------------------------
# 4) A priori covariance P0
# -----------------------------
sigma_r0_km = 1.0
sigma_v0_km_s = 1.0e-3
P0 = np.diag([
    sigma_r0_km**2, sigma_r0_km**2, sigma_r0_km**2,
    sigma_v0_km_s**2, sigma_v0_km_s**2, sigma_v0_km_s**2
])

# -----------------------------
# 5) Process noise Q
# -----------------------------
Q = np.zeros((6, 6), dtype=float)

# -----------------------------
# 6) Truth + initial guess
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

X0_star = x0_true + dx

truth_times = truth_df["t_s"].to_numpy(float)
truth_states = truth_df[["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]].to_numpy(float)
Xtrue_meas = np.column_stack([np.interp(t_meas, truth_times, truth_states[:,k]) for k in range(6)])

# -----------------------------
# 7) Dynamics parameters
# -----------------------------
mu = 398600.4418
J2 = 1.08262668e-3
J3 = -2.532153e-6

# -----------------------------
# 8) Warmstart: LKF for N_boot, then EKF
# -----------------------------
N_boot = 100  # <-- choose whatever you want

out = warmstart_ekf_with_lkf(
    all_meas=all_meas,
    stations=stations,
    Xtrue_meas=Xtrue_meas,
    N_boot=N_boot,
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

print("\n=== Warmstart (LKF -> EKF) finished ===")
print("Xhat_meas shape:", out["Xhat_meas"].shape)
print("postfit_resids_meas shape:", out["postfit_resids_meas"].shape)

# -----------------------------
# 9) Convenience handles
# -----------------------------
t = out["t_meas"]
err = out["state_error_meas"]
post = out["postfit_resids_meas"]
two_sigma = out["two_sigma_meas"]
labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]

# EKF-only portion mask (everything after the bootstrap)
mask_ekf = np.zeros_like(t, dtype=bool)
mask_ekf[N_boot:] = True

t_ekf = t[mask_ekf]
err_ekf = None if err is None else err[mask_ekf, :]
post_ekf = post[mask_ekf, :]
two_sigma_ekf = two_sigma[mask_ekf, :]

# -----------------------------
# 10) Plot: FULL timeline state error (±3σ state bounds)
# -----------------------------
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()

three_sigma_state = (3.0 / 2.0) * two_sigma  # two_sigma = 2σ, so multiply by 3/2

for i in range(6):
    axs[i].scatter(t, err[:, i], s=6, marker="o", label="state error" if i == 0 else None)
    axs[i].plot(t, +three_sigma_state[:, i], linestyle="--", linewidth=1.5, color="k",
                label="+3σ bound" if i == 0 else None)
    axs[i].plot(t, -three_sigma_state[:, i], linestyle="--", linewidth=1.5, color="k",
                label="-3σ bound" if i == 0 else None)
    axs[i].set_ylabel(labels[i])
    axs[i].grid(True)
    axs[i].axvline(t[N_boot - 1], linestyle=":", linewidth=2, color="k")  # handoff marker

axs[-2].set_xlabel("t (s)")
axs[-1].set_xlabel("t (s)")
fig.suptitle(f"Warmstart State Error vs Time (LKF first {N_boot}, then EKF)", y=0.98)
fig.legend(loc="upper right")
plt.tight_layout()

# -----------------------------
# 11) Plot: FULL timeline postfit residuals (±3σ measurement bounds)
# -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axs[0].scatter(t, post[:, 0], s=6, marker="o", label="postfit residual")
axs[0].axhline(0.0, linewidth=1)
axs[0].axhline(+3 * sigma_rho_km, linestyle="--", linewidth=1.5, color="k", label="+3σ meas bound")
axs[0].axhline(-3 * sigma_rho_km, linestyle="--", linewidth=1.5, color="k", label="-3σ meas bound")
axs[0].axvline(t[N_boot - 1], linestyle=":", linewidth=2, color="k")
axs[0].set_ylabel("Postfit ρ residual (km)")
axs[0].grid(True)
axs[0].legend(loc="upper right")

axs[1].scatter(t, post[:, 1], s=6, marker="o", label="postfit residual")
axs[1].axhline(0.0, linewidth=1)
axs[1].axhline(+3 * sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k", label="+3σ meas bound")
axs[1].axhline(-3 * sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k", label="-3σ meas bound")
axs[1].axvline(t[N_boot - 1], linestyle=":", linewidth=2, color="k")
axs[1].set_ylabel("Postfit ρ̇ residual (km/s)")
axs[1].set_xlabel("t (s)")
axs[1].grid(True)
axs[1].legend(loc="upper right")

fig.suptitle("Warmstart Postfit Residuals vs Time (full timeline)", y=0.98)
plt.tight_layout()

# -----------------------------
# 12) Plot: EKF-only state error (±3σ state bounds)
# -----------------------------
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()

three_sigma_state_ekf = (3.0 / 2.0) * two_sigma_ekf

for i in range(6):
    axs[i].scatter(t_ekf, err_ekf[:, i], s=6, marker="o", label="state error" if i == 0 else None)
    axs[i].plot(t_ekf, +three_sigma_state_ekf[:, i], linestyle="--", linewidth=1.5, color="k",
                label="+3σ bound" if i == 0 else None)
    axs[i].plot(t_ekf, -three_sigma_state_ekf[:, i], linestyle="--", linewidth=1.5, color="k",
                label="-3σ bound" if i == 0 else None)
    axs[i].set_ylabel(labels[i])
    axs[i].grid(True)

axs[-2].set_xlabel("t (s)")
axs[-1].set_xlabel("t (s)")
fig.suptitle("EKF-only State Error vs Time (after warmstart)", y=0.98)
fig.legend(loc="upper right")
plt.tight_layout()

# -----------------------------
# 13) Plot: EKF-only postfit residuals (±3σ measurement bounds)
# -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axs[0].scatter(t_ekf, post_ekf[:, 0], s=6, marker="o", label="postfit residual")
axs[0].axhline(0.0, linewidth=1)
axs[0].axhline(+3 * sigma_rho_km, linestyle="--", linewidth=1.5, color="k", label="+3σ meas bound")
axs[0].axhline(-3 * sigma_rho_km, linestyle="--", linewidth=1.5, color="k", label="-3σ meas bound")
axs[0].set_ylabel("Postfit ρ residual (km)")
axs[0].grid(True)
axs[0].legend(loc="upper right")

axs[1].scatter(t_ekf, post_ekf[:, 1], s=6, marker="o", label="postfit residual")
axs[1].axhline(0.0, linewidth=1)
axs[1].axhline(+3 * sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k", label="+3σ meas bound")
axs[1].axhline(-3 * sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k", label="-3σ meas bound")
axs[1].set_ylabel("Postfit ρ̇ residual (km/s)")
axs[1].set_xlabel("t (s)")
axs[1].grid(True)
axs[1].legend(loc="upper right")

fig.suptitle("EKF-only Postfit Residuals vs Time (after warmstart)", y=0.98)
plt.tight_layout()

plt.show()
