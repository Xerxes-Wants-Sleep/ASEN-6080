import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("../../")

from src.Functions.stations import Stations
from src.Functions.filters import ExtendedKalmanFilter
from src.Functions.range_rangerate import H_range_rangerate
from src.Functions.snc import state_noise_compensation


# -----------------------------
# Problem 1 (EKF + SNC sweep)
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MEAS_PATH = SCRIPT_DIR.parent / "Homework2" / "meas_data" / "prob2_hw2_measurements_noisy.csv"
TRUTH_PATH = SCRIPT_DIR.parent / "Homework1" / "HW2_j3_on_truth.csv"

PLOTS_DIR = SCRIPT_DIR / "Plots" / "EKF_SNC"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Read measurements
# -----------------------------
df = pd.read_csv(MEAS_PATH)
all_meas = sorted(df.to_dict(orient="records"), key=lambda m: float(m["t"]))
t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)

# -----------------------------
# 2) Stations
# -----------------------------
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944, theta0_deg=122, radius_earth=6378.0, w_earth_rad_per_s=2*np.pi/86400),
    Stations("Station 2", lat_deg=40.427222, lon_deg=355.749444, theta0_deg=122, radius_earth=6378.0, w_earth_rad_per_s=2*np.pi/86400),
    Stations("Station 3", lat_deg=35.247164, lon_deg=243.205000, theta0_deg=122, radius_earth=6378.0, w_earth_rad_per_s=2*np.pi/86400),
]


def build_callbacks(stations_in):
    station_map = {st.name: st for st in stations_in}

    def get_measurement(m):
        return np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)

    def predict_obs(x, m):
        st = station_map[m["station"]]
        d = st.measure(x[0:3], x[3:6], float(m["t"]))
        if d is None:
            return None
        return np.array([d["rho"], d["rho_dot"]], dtype=float)

    def H_matrix(x, m):
        st = station_map[m["station"]]
        Rs, Vs, _ = st.ecef2eci(float(m["t"]), st.r_ecef, np.zeros(3))
        H_sc = H_range_rangerate(x[0:3], x[3:6], Rs, Vs)
        H = np.zeros((2, x.size), dtype=float)
        H[:, 0:3] = H_sc[:, 0:3]
        H[:, 3:6] = H_sc[:, 3:6]
        return H

    return get_measurement, predict_obs, H_matrix


get_measurement, predict_obs, H_matrix = build_callbacks(stations)

# -----------------------------
# 3) Noise + initial covariance
# -----------------------------
sigma_rho_km = 1.0e-3
sigma_rhod_km_s = 1.0e-6
R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

sigma_r0_km = 1.0e3
sigma_v0_km_s = 1.0
P0 = np.diag(
    [
        sigma_r0_km**2,
        sigma_r0_km**2,
        sigma_r0_km**2,
        sigma_v0_km_s**2,
        sigma_v0_km_s**2,
        sigma_v0_km_s**2,
    ]
)

# -----------------------------
# 4) Truth + initial guess
# -----------------------------
truth_df = pd.read_csv(TRUTH_PATH)
truth_times = truth_df["t_s"].to_numpy(float)
truth_states = truth_df[["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
Xtrue_meas = np.column_stack([np.interp(t_meas, truth_times, truth_states[:, k]) for k in range(6)])

dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
x0_true = truth_states[0, :]
x0_guess = x0_true + dx

# -----------------------------
# 5) Dynamics setup
# -----------------------------
mu = 398600.4415
J2 = 0.0010826269
J3 = -2.5324e-6

# -----------------------------
# 6) Sigma sweep setup
# -----------------------------
# sigma in km/s^2 (matches MATLAB script units)
sigma_grid_kmps2 = np.logspace(-15, -2, 14)


def make_snc_q(sigma_kmps2):
    return (float(sigma_kmps2) ** 2) * np.eye(3, dtype=float)


def snc_trace_series_for_sigma(sigma_kmps2):
    q_acc = make_snc_q(sigma_kmps2)
    dt_steps = np.diff(t_meas, prepend=t_meas[0]).astype(float)
    tr = np.zeros_like(dt_steps, dtype=float)
    for i, dt in enumerate(dt_steps):
        qk = state_noise_compensation(float(dt), 6, 3, q_acc)
        tr[i] = float(np.trace(qk))
    return dt_steps, tr


def snc_trace_stats_for_sigma(sigma_kmps2):
    _, tr = snc_trace_series_for_sigma(sigma_kmps2)
    return {
        "snc_trace_mean": float(np.mean(tr)),
        "snc_trace_max": float(np.max(tr)),
        "snc_trace_total": float(np.sum(tr)),
    }


def run_once(sigma_kmps2):
    Q = make_snc_q(sigma_kmps2)
    ekf = ExtendedKalmanFilter(
        x0=x0_guess,
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
        first_pass_gap_s=6 * 3600.0,
        state_mapping_dict={"pos_idx": [0, 1, 2], "vel_idx": [3, 4, 5]},
    )
    return ekf.run(
        all_meas=all_meas,
        get_measurement=get_measurement,
        predict_obs=predict_obs,
        H_matrix=H_matrix,
        Xtrue_meas=Xtrue_meas,
    )


def save_fig(fig, filename):
    fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

# quick cadence diagnostic: dt vs t
dt_series = np.diff(t_meas, prepend=t_meas[0]).astype(float)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_meas, dt_series, ".", markersize=2)
ax.set_xlabel("t (s)")
ax.set_ylabel("dt (s)")
ax.set_title("Measurement cadence: dt vs t (EKF)")
ax.grid(True)
fig.tight_layout()
save_fig(fig, "dt_vs_t.png")


# -----------------------------
# 7) Run sigma sweep and store metrics
# -----------------------------
rows = []
for s in sigma_grid_kmps2:
    out = run_once(s)
    rms = out["rms_final"]
    pf_lin = np.asarray(out["postfit_resids_linear_final"], dtype=float)
    if pf_lin.size == 0 or not np.isfinite(pf_lin).any():
        raise RuntimeError(
            "Linearized postfit RMS is NaN. This usually means measurement callbacks are rejecting all measurements "
            "(station geometry/time settings do not match the measurement file)."
        )
    rms_postfit_lin = np.sqrt(np.nanmean(pf_lin**2, axis=0))
    pf_norm2 = np.sum(pf_lin**2, axis=1)
    postfit_total = float(np.sqrt(np.nanmean(pf_norm2)))
    rows.append(
        {
            "sigma_kmps2": float(s),
            "postfit_rho_rms_km": float(rms_postfit_lin[0]),
            "postfit_rhod_rms_km_s": float(rms_postfit_lin[1]),
            "postfit_total_rms": postfit_total,
            "pos3_rms_km": float(rms["pos3_all"]),
            "vel3_rms_km_s": float(rms["vel3_all"]),
            **snc_trace_stats_for_sigma(s),
        }
    )
    print(
        f"sigma={s:.3e} | postfit(rho,rhod)=({rows[-1]['postfit_rho_rms_km']:.3e}, "
        f"{rows[-1]['postfit_rhod_rms_km_s']:.3e}) | 3D(pos,vel)=({rows[-1]['pos3_rms_km']:.3e}, {rows[-1]['vel3_rms_km_s']:.3e})"
    )

df_metrics = pd.DataFrame(rows).sort_values("sigma_kmps2").reset_index(drop=True)
df_metrics.to_csv(PLOTS_DIR / "sigma_sweep_metrics.csv", index=False)

# -----------------------------
# 8) Plot metrics vs sigma
# -----------------------------
sig = df_metrics["sigma_kmps2"].to_numpy(float)

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(sig, df_metrics["postfit_rho_rms_km"], "o-", label="rho RMS (km)")
ax.loglog(sig, df_metrics["postfit_rhod_rms_km_s"], "s-", label="rho_dot RMS (km/s)")
ax.loglog(sig, df_metrics["postfit_total_rms"], "^-", label="combined linearized postfit RMS")
ax.set_xlabel("sigma (km/s^2)")
ax.set_ylabel("linearized postfit RMS")
ax.set_title("EKF + SNC: Linearized Postfit RMS vs sigma")
ax.grid(True, which="both")
ax.legend()
fig.tight_layout()
save_fig(fig, "postfit_rms_vs_sigma.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(sig, df_metrics["pos3_rms_km"], "o-", label="3D position RMS (km)")
ax.loglog(sig, df_metrics["vel3_rms_km_s"], "s-", label="3D velocity RMS (km/s)")
ax.set_xlabel("sigma (km/s^2)")
ax.set_ylabel("3D RMS")
ax.set_title("EKF + SNC: 3D RMS vs sigma")
ax.grid(True, which="both")
ax.legend()
fig.tight_layout()
save_fig(fig, "state3d_rms_vs_sigma.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(sig, df_metrics["snc_trace_mean"], "o-", label="mean trace(Qk_snc)")
ax.loglog(sig, df_metrics["snc_trace_max"], "s-", label="max trace(Qk_snc)")
ax.loglog(sig, df_metrics["snc_trace_total"], "^-", label="sum trace(Qk_snc)")
ax.set_xlabel("sigma (km/s^2)")
ax.set_ylabel("SNC covariance trace")
ax.set_title("EKF + SNC: Injected SNC Covariance vs sigma")
ax.grid(True, which="both")
ax.legend()
fig.tight_layout()
save_fig(fig, "snc_trace_vs_sigma.png")

sigma_probe = 1.0e-8
dt_probe, tr_probe = snc_trace_series_for_sigma(sigma_probe)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(t_meas, tr_probe, ".", markersize=2)
ax1.set_ylabel("trace(Qk_snc)")
ax1.set_title(f"EKF + SNC: Injected SNC vs time (sigma={sigma_probe:.1e} km/s^2)")
ax1.grid(True)
ax2.plot(t_meas, dt_probe, ".", markersize=2)
ax2.set_ylabel("dt (s)")
ax2.set_xlabel("t (s)")
ax2.grid(True)
fig.tight_layout()
save_fig(fig, "snc_vs_time_sigma_1e-8.png")

# -----------------------------
# 9) Pick "optimal sigma" and rerun
# -----------------------------
eps = 1e-30
objective = (
    df_metrics["postfit_rho_rms_km"] / (df_metrics["postfit_rho_rms_km"].min() + eps)
    + df_metrics["postfit_rhod_rms_km_s"] / (df_metrics["postfit_rhod_rms_km_s"].min() + eps)
    + df_metrics["pos3_rms_km"] / (df_metrics["pos3_rms_km"].min() + eps)
    + df_metrics["vel3_rms_km_s"] / (df_metrics["vel3_rms_km_s"].min() + eps)
)
idx_best = int(np.argmin(objective.to_numpy(float)))
sigma_opt = float(df_metrics.loc[idx_best, "sigma_kmps2"])
print(f"\nEKF optimal sigma (balanced metric): {sigma_opt:.6e} km/s^2")

out_opt = run_once(sigma_opt)
t = out_opt["t_meas"]
err = out_opt["state_error_meas"]
post = out_opt["postfit_resids_linear_final"]

# -----------------------------
# 10) Plot time history at optimal sigma
# -----------------------------
labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()
for i in range(6):
    axs[i].plot(t, err[:, i], linewidth=1.0, label="state error")
    axs[i].set_ylabel(labels[i])
    axs[i].grid(True)

if "two_sigma_meas" in out_opt and out_opt["two_sigma_meas"] is not None:
    three_sigma = (3.0 / 2.0) * np.asarray(out_opt["two_sigma_meas"], dtype=float)
    for i in range(6):
        axs[i].plot(t, +three_sigma[:, i], "k--", linewidth=1.2, label="+3sigma")
        axs[i].plot(t, -three_sigma[:, i], "k--", linewidth=1.2, label="-3sigma")

axs[-2].set_xlabel("t (s)")
axs[-1].set_xlabel("t (s)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right")
fig.suptitle(f"EKF + SNC: State Error at Optimal sigma = {sigma_opt:.3e} km/s^2", y=0.98)
fig.tight_layout()
save_fig(fig, "optimal_state_error_time_history.png")

fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axs[0].plot(t, post[:, 0], linewidth=1.0)
axs[0].axhline(0.0, linewidth=1.0, color="k")
axs[0].axhline(+3 * sigma_rho_km, linestyle="--", linewidth=1.2, color="k")
axs[0].axhline(-3 * sigma_rho_km, linestyle="--", linewidth=1.2, color="k")
axs[0].set_ylabel("Linearized postfit rho (km)")
axs[0].grid(True)

axs[1].plot(t, post[:, 1], linewidth=1.0)
axs[1].axhline(0.0, linewidth=1.0, color="k")
axs[1].axhline(+3 * sigma_rhod_km_s, linestyle="--", linewidth=1.2, color="k")
axs[1].axhline(-3 * sigma_rhod_km_s, linestyle="--", linewidth=1.2, color="k")
axs[1].set_ylabel("Linearized postfit rho_dot (km/s)")
axs[1].set_xlabel("t (s)")
axs[1].grid(True)

fig.suptitle("EKF + SNC: Linearized Postfit Residuals at Optimal sigma", y=0.98)
fig.tight_layout()
save_fig(fig, "optimal_postfit_time_history.png")

print(f"\nSaved EKF SNC outputs to: {PLOTS_DIR.resolve()}")

