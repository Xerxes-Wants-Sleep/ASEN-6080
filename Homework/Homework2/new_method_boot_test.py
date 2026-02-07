import numpy as np
import sys
import pandas as pd
from pathlib import Path

# If running from Homework2 folder, this makes `src/` importable
sys.path.append("../../")

from src.Functions.stations import Stations
from src.Functions.filters import LinearizedKalmanFilter, ExtendedKalmanFilter

from src.helpers.plotting.post_processing import run_filter_post_processing, print_rms_summary

from src.helpers.plotting.plot_prefit_residuals import make_prefit_residuals_plot
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_postfit_residuals_nonlinear import make_postfit_residuals_nonlinear_plot
from src.helpers.plotting.plot_state_errors_eci import make_state_errors_eci_plots
from src.helpers.plotting.plot_state_errors_rsw import make_state_errors_rsw_plot
from src.helpers.plotting.plot_cov_diag_log import make_cov_diag_log_plot
from src.helpers.plotting.plot_trace_cov import make_trace_cov_plot


# -----------------------------
# Load measurements
# -----------------------------
df = pd.read_csv("meas_data/hw2_measurements_noisy.csv")
all_meas = df.to_dict(orient="records")

# sort once so truth alignment matches the filter split
all_meas_sorted = sorted(all_meas, key=lambda m: float(m["t"]))
t_meas = np.array([float(m["t"]) for m in all_meas_sorted], dtype=float)

# -----------------------------
# Stations
# -----------------------------
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222, lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164, lon_deg=243.205000),
]

# -----------------------------
# Measurement noise
# -----------------------------
sigma_rho_km = 1e-3
sigma_rhod_km_s = 1e-6
R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

# -----------------------------
# A priori covariance + process noise
# -----------------------------
P0 = np.diag([1.0**2]*3 + [1e-3**2]*3)
Q = np.zeros((6, 6))

# -----------------------------
# Truth aligned to measurement times
# -----------------------------
truth_df = pd.read_csv("../Homework1/prob2a_traj.csv")
truth_times = truth_df["t_s"].to_numpy(float)
truth_states = truth_df[["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]].to_numpy(float)

Xtrue_meas = np.column_stack([
    np.interp(t_meas, truth_times, truth_states[:, k]) for k in range(6)
])

x0_true = truth_states[0, :]
dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
X0_star = x0_true + dx

# -----------------------------
# Dynamics params
# -----------------------------
mu = 398600.4415
J2 = 0.0010826269
J3 = -2.5324e-6

# -----------------------------
# Build filters
# -----------------------------
lkf = LinearizedKalmanFilter(
    X0_star=X0_star, P0=P0, R=R, Q=Q,
    mu=mu, J2=J2, J3=J3,
    j2=True, j3=False,
    first_pass_gap_s=6*3600.0,
)

ekf = ExtendedKalmanFilter(
    x0=X0_star, P0=P0, R=R, Q=Q,
    mu=mu, J2=J2, J3=J3,
    j2=True, j3=False,
    first_pass_gap_s=6*3600.0,
)

# -----------------------------
# Warmstart run
# -----------------------------
NUM_INIT_MEAS = 100

out = ekf.run_warmstarted(
    all_meas=all_meas_sorted,   # already sorted
    stations=stations,
    lkf=lkf,
    num_init_meas=NUM_INIT_MEAS,
    Xtrue_meas=Xtrue_meas,      # MUST align to same sorted order
)

print("\n=== Warmstart finished ===")
print("t_start_ekf:", out.get("t_start_ekf", None))
print("Xhat_meas shape:", out["Xhat_meas"].shape)
print("P_meas shape:", out["P_meas"].shape)
print("postfit_resids_meas shape:", out["postfit_resids_meas"].shape)

# -----------------------------
# Convert to BatchPostProcessResult so batch plotters work
# -----------------------------
result = run_filter_post_processing(out=out)

print_rms_summary(result, ignore_first_pass=False)
print_rms_summary(result, ignore_first_pass=True)

# -----------------------------
# Plots (batch-style)
# -----------------------------
PLOT_DIR = Path("Plots") / f"Warmstart_LKF{NUM_INIT_MEAS}_EKF"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

make_prefit_residuals_plot(result, PLOT_DIR)
make_postfit_residuals_linear_plot(result, PLOT_DIR)
make_postfit_residuals_nonlinear_plot(result, PLOT_DIR)   
make_state_errors_eci_plots(result, PLOT_DIR)
make_state_errors_rsw_plot(result, PLOT_DIR)
make_cov_diag_log_plot(result, PLOT_DIR)
make_trace_cov_plot(result, PLOT_DIR)

print(f"\nSaved plots to: {PLOT_DIR.resolve()}")
