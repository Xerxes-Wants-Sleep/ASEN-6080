import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("../../")

from src.Functions.stations import Stations
from src.Functions.filters import ExtendedKalmanFilter, LinearizedKalmanFilter

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

# sigma_r0_km = 1.0e3 
# sigma_v0_km_s = 1.0       # 1 km/s


P0 = np.diag([
    sigma_r0_km**2, sigma_r0_km**2, sigma_r0_km**2,
    sigma_v0_km_s**2, sigma_v0_km_s**2, sigma_v0_km_s**2
])

SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "Plots" / "EKF WS B"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, filename, dpi=300):
    """Save and close a matplotlib figure to PLOTS_DIR."""
    outpath = PLOTS_DIR / filename
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# -----------------------------
# 5) Process noise Q
# -----------------------------
Q = np.zeros((6, 6), dtype=float)

# -----------------------------
# 6) Truth + initial guess
# -----------------------------
dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
# dx = 100 * np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)



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
Xtrue_meas = np.column_stack([np.interp(t_meas, truth_times, truth_states[:, k]) for k in range(6)])

# -----------------------------
# 7) Dynamics parameters
# -----------------------------
mu = 398600.4415
J2 = 0.0010826269
J3 = -2.5324e-6

# -----------------------------
# 8) Warmstart settings
# -----------------------------
N_boot_nominal = 1000      # requested bootstrap count (method may extend to end-of-epoch)

# -----------------------------
# 9) Run warmstarted EKF (LKF -> EKF)
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

ekf = ExtendedKalmanFilter(
    x0=X0_star,
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

# NOTE: method name depends on what you renamed it to.
# If you renamed it to `run_warmstarted`, use that.
out = ekf.run_warmstarted(
    all_meas=all_meas,
    stations=stations,
    lkf=lkf,
    num_init_meas=N_boot_nominal,
    Xtrue_meas=Xtrue_meas,
)

print("\n=== Warmstarted EKF finished (LKF -> EKF) ===")
print("Xhat_meas shape:", out["Xhat_meas"].shape)
print("postfit_resids_meas shape:", out["postfit_resids_meas"].shape)

# handoff index/time
N_boot_used = len(out["lkf_init"]["t_meas"])
t_handoff = float(out["t_start_ekf"])


# -----------------------------
# Extract data
# -----------------------------
t = np.asarray(out["t_meas"], dtype=float)
err = np.asarray(out["state_error_meas"], dtype=float)
post = np.asarray(out["postfit_resids_meas"], dtype=float)
two_sigma = np.asarray(out["two_sigma_meas"], dtype=float)

labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]


# -----------------------------
# 1) State error vs time (±3σ) + handoff line
# -----------------------------
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()

sc_handle = None
bound_handle = None

for i in range(6):
    ax = axs[i]
    sc = ax.scatter(t, err[:, i], s=10, marker="o")
    if sc_handle is None:
        sc_handle = sc

    # handoff
    ax.axvline(t_handoff, linestyle=":", linewidth=2, color="k")

    # ±3σ bounds
    three_sigma = 1.5 * two_sigma[:, i]  # two_sigma is 2σ -> multiply by 1.5 for 3σ
    h1, = ax.plot(t, +three_sigma, linestyle="--", linewidth=1.5, color="k")
    ax.plot(t, -three_sigma, linestyle="--", linewidth=1.5, color="k")
    if bound_handle is None:
        bound_handle = h1

    ax.set_ylabel(labels[i])
    ax.grid(True)

axs[-2].set_xlabel("t (s)")
axs[-1].set_xlabel("t (s)")

handoff_handle = axs[0].axvline(t_handoff, linestyle=":", linewidth=2, color="k")
fig.legend(
    [sc_handle, bound_handle, handoff_handle],
    ["state error", "±3σ bound", "LKF→EKF handoff"],
    loc="upper right"
)

fig.suptitle("Warmstart (LKF → EKF) State Error vs Time (±3σ)", y=0.98)
plt.tight_layout()

save_fig(fig, "warmstart_state_error_all.png")


# -----------------------------
# 2) Postfit residuals vs time (±3σ noise) + handoff line
# -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

sc0 = axs[0].scatter(t, post[:, 0], s=10, marker="o")
axs[0].axhline(0.0, linewidth=1)
n0p = axs[0].axhline(+3 * sigma_rho_km, linestyle="--", linewidth=1.5, color="k")
axs[0].axhline(-3 * sigma_rho_km, linestyle="--", linewidth=1.5, color="k")
handoff0 = axs[0].axvline(t_handoff, linestyle=":", linewidth=2, color="k")
axs[0].set_ylabel("Postfit ρ residual (km)")
axs[0].grid(True)

axs[1].scatter(t, post[:, 1], s=10, marker="o")
axs[1].axhline(0.0, linewidth=1)
axs[1].axhline(+3 * sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k")
axs[1].axhline(-3 * sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k")
axs[1].axvline(t_handoff, linestyle=":", linewidth=2, color="k")
axs[1].set_ylabel("Postfit ρ̇ residual (km/s)")
axs[1].set_xlabel("t (s)")
axs[1].grid(True)

fig.legend(
    [sc0, n0p, handoff0],
    ["postfit residual", "±3σ meas noise", "LKF→EKF handoff"],
    loc="upper right"
)

fig.suptitle("Warmstart (LKF → EKF) Postfit Residuals vs Time (±3σ)", y=0.98)
plt.tight_layout()

save_fig(fig, "warmstart_postfit_residuals.png")


# -----------------------------
# 3) Zoomed state error vs time (±3σ) + handoff line
# -----------------------------
t0_zoom = 40000.0
mask = t >= t0_zoom

fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()

sc_handle = None
bound_handle = None

for i in range(6):
    ax = axs[i]
    sc = ax.scatter(t[mask], err[mask, i], s=10, marker="o")
    if sc_handle is None:
        sc_handle = sc

    ax.axvline(t_handoff, linestyle=":", linewidth=2, color="k")

    three_sigma = 1.5 * two_sigma[mask, i]
    h1, = ax.plot(t[mask], +three_sigma, linestyle="--", linewidth=1.5, color="k")
    ax.plot(t[mask], -three_sigma, linestyle="--", linewidth=1.5, color="k")
    if bound_handle is None:
        bound_handle = h1

    ax.set_ylabel(labels[i])
    ax.grid(True)

axs[-2].set_xlabel("t (s)")
axs[-1].set_xlabel("t (s)")

handoff_handle = axs[0].axvline(t_handoff, linestyle=":", linewidth=2, color="k")
fig.legend(
    [sc_handle, bound_handle, handoff_handle],
    ["state error", "±3σ bound", "LKF→EKF handoff"],
    loc="upper right"
)

fig.suptitle(f"Warmstart (LKF → EKF) State Error vs Time (Zoom: t ≥ {t0_zoom:.0f} s)", y=0.98)
plt.tight_layout()

save_fig(fig, f"warmstart_state_error_zoom_t_ge_{int(t0_zoom)}.png")

