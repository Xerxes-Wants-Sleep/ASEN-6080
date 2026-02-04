import numpy as np
import sys
import pandas as pd

from pathlib import Path
# import matplotlib
# matplotlib.use("Agg")
sys.path.append("../../")

from src.Functions.stations import Stations
from src.Functions.batch import batch_estimate_x0

from src.helpers.plotting.post_processing import run_batch_post_processing, print_rms_summary
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
meas_path = "meas_data/hw2_measurements_noisy.csv"
df = pd.read_csv(meas_path)
all_meas = df.to_dict(orient="records")

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
sigma_rho_km = 1.0e-3
sigma_rhod_km_s = 1.0e-6
R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

# -----------------------------
# A priori covariance
# -----------------------------
sigma_r0_km = 1.0
sigma_v0_km_s = 1.0e-3
P0 = np.diag([sigma_r0_km**2]*3 + [sigma_v0_km_s**2]*3)

# -----------------------------
# Initial guess (truth + dx)
# -----------------------------
dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)

truth_path = "../Homework1/prob2a_traj.csv"
truth_df = pd.read_csv(truth_path)

x0_true = np.array(
    [truth_df.loc[0, c] for c in ["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]],
    dtype=float
)
x0_bar = x0_true + dx

truth_times = truth_df["t_s"].to_numpy(float)
truth_states_6 = truth_df[["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]].to_numpy(float)

# -----------------------------
# Dynamics params
# -----------------------------
mu = 398600.4415
J2 = 0.0010826269
J3 = -2.5324e-6

# -----------------------------
# Run batch
# -----------------------------
x0_hat, P0_hat, info = batch_estimate_x0(
    all_meas=all_meas,
    stations=stations,
    x0_bar=x0_bar,
    P0=P0,
    R=R,
    mu=mu,
    J2=J2,
    J3=False,
    max_iter=10,
    tol=1e-10,
    reltol=1e-10,
    abstol=1e-10,
)

print("\n=== Batch finished ===")
print("num_iters:", info["num_iters"])
print("x0_hat:", x0_hat)
print("diag(P0_hat):", np.diag(P0_hat))

# -----------------------------
# Post-process + RMS prints
# -----------------------------
result = run_batch_post_processing(
    all_meas=all_meas,
    stations=stations,
    x0_hat=x0_hat,
    P0_hat=P0_hat,
    mu=mu,
    J2=J2,
    J3=J3,
    truth_times=truth_times,
    truth_states_6=truth_states_6,
    info=info,                 # gives prefit + linear postfit arrays
    x0_star_hist=None,         # set to info["x0_star_hist"] if you want RMS vs iter
    first_pass_gap_s=6*3600.0,
)

print_rms_summary(result, ignore_first_pass=False)
print_rms_summary(result, ignore_first_pass=True)

# -----------------------------
# Make plots
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "Plots" / "Batch"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

make_prefit_residuals_plot(result, PLOT_DIR)
make_postfit_residuals_linear_plot(result, PLOT_DIR)
make_postfit_residuals_nonlinear_plot(result, PLOT_DIR)

make_state_errors_eci_plots(result, PLOT_DIR)
make_state_errors_rsw_plot(result, PLOT_DIR)

make_cov_diag_log_plot(result, PLOT_DIR)
make_trace_cov_plot(result, PLOT_DIR)

print(f"\nSaved plots to: {PLOT_DIR}")
