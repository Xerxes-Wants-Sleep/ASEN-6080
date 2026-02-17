import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("../../")

from src.Functions.stations import Stations
from src.Functions.filters import LinearizedKalmanFilter
from src.Functions.range_rangerate import H_range_rangerate
from src.Functions.snc import state_noise_compensation
from src.helpers.plotting.post_processing import run_filter_post_processing, print_rms_summary
from src.helpers.plotting.plot_prefit_residuals import make_prefit_residuals_plot
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_postfit_residuals_nonlinear import make_postfit_residuals_nonlinear_plot
from src.helpers.plotting.plot_state_errors_eci import make_state_errors_eci_plots
from src.helpers.plotting.plot_state_errors_rsw import make_state_errors_rsw_plot
from src.helpers.plotting.plot_cov_diag_log import make_cov_diag_log_plot
from src.helpers.plotting.plot_trace_cov import make_trace_cov_plot


# -----------------------------
# Problem 1 (LKF + SNC sweep)
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MEAS_PATH = SCRIPT_DIR.parent / "Homework2" / "meas_data" / "prob2_hw2_measurements_noisy.csv"
TRUTH_PATH = SCRIPT_DIR.parent / "Homework1" / "HW2_j3_on_truth.csv"

PLOTS_DIR = SCRIPT_DIR / "Plots" / "LKF_SNC"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Read measurements
# -----------------------------
df = pd.read_csv(MEAS_PATH)
all_meas = df.to_dict(orient="records")
all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)

# ---- stations ----
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

dx = 100 * np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)

x0_true = truth_states[0, :]
X0_star = x0_true + dx

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
        first_pass_gap_s=6 * 3600.0,
        state_mapping_dict={"pos_idx": [0, 1, 2], "vel_idx": [3, 4, 5]},
    )
    return lkf.run(
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
ax.set_title("Measurement cadence: dt vs t (LKF)")
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
ax.set_title("LKF + SNC: Linearized Postfit RMS vs sigma")
ax.grid(True, which="both")
ax.legend()
fig.tight_layout()
save_fig(fig, "postfit_rms_vs_sigma.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(sig, df_metrics["pos3_rms_km"], "o-", label="3D position RMS (km)")
ax.loglog(sig, df_metrics["vel3_rms_km_s"], "s-", label="3D velocity RMS (km/s)")
ax.set_xlabel("sigma (km/s^2)")
ax.set_ylabel("3D RMS")
ax.set_title("LKF + SNC: 3D RMS vs sigma")
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
ax.set_title("LKF + SNC: Injected SNC Covariance vs sigma")
ax.grid(True, which="both")
ax.legend()
fig.tight_layout()
save_fig(fig, "snc_trace_vs_sigma.png")

sigma_probe = 1.0e-8
dt_probe, tr_probe = snc_trace_series_for_sigma(sigma_probe)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(t_meas, tr_probe, ".", markersize=2)
ax1.set_ylabel("trace(Qk_snc)")
ax1.set_title(f"LKF + SNC: Injected SNC vs time (sigma={sigma_probe:.1e} km/s^2)")
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
print(f"\nLKF optimal sigma (balanced metric): {sigma_opt:.6e} km/s^2")

out_opt = run_once(sigma_opt)
# -----------------------------
# 10) Reuse existing plotting code at optimal sigma
# -----------------------------
result_opt = run_filter_post_processing(out=out_opt)
print_rms_summary(result_opt, ignore_first_pass=False)
print_rms_summary(result_opt, ignore_first_pass=True)

PLOT_DIR_OPT = PLOTS_DIR / "optimal_sigma_fullplots"
PLOT_DIR_OPT.mkdir(parents=True, exist_ok=True)
make_prefit_residuals_plot(result_opt, PLOT_DIR_OPT)
make_postfit_residuals_linear_plot(result_opt, PLOT_DIR_OPT)
make_postfit_residuals_nonlinear_plot(result_opt, PLOT_DIR_OPT)
make_state_errors_eci_plots(result_opt, PLOT_DIR_OPT)
make_state_errors_rsw_plot(result_opt, PLOT_DIR_OPT)
make_cov_diag_log_plot(result_opt, PLOT_DIR_OPT)
make_trace_cov_plot(result_opt, PLOT_DIR_OPT)

print(f"\nSaved LKF SNC outputs to: {PLOTS_DIR.resolve()}")
