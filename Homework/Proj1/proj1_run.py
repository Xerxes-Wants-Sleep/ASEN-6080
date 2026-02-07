import numpy as np
import sys
from dataclasses import replace

from pathlib import Path
sys.path.append("../../")

from src.Functions.batch_18_state import batch_estimate_x0
from src.Functions.dynamics_muJ2_drag import f_muJ2_drag, A_muJ2_drag
from src.Functions.propagation import PropSettings

from src.helpers.plotting.post_processing import run_batch_post_processing_18, print_rms_summary
from src.helpers.plotting.plot_prefit_residuals import make_prefit_residuals_plot
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_postfit_residuals_nonlinear import make_postfit_residuals_nonlinear_plot
from src.helpers.plotting.plot_cov_diag_log import make_cov_diag_log_plot
from src.helpers.plotting.plot_trace_cov import make_trace_cov_plot
from src.helpers.plotting.plot_state_errors_eci import make_state_errors_eci_plots
from src.helpers.plotting.plot_state_errors_rsw import make_state_errors_rsw_plot


# -----------------------------
# Load measurements
# -----------------------------
meas_path = Path(__file__).resolve().parent / "Given_Data" / "project.txt"
data = np.loadtxt(meas_path)

t_meas = data[:, 0]
station_id = data[:, 1].astype(int)
rho_m = data[:, 2]
rho_dot_m_s = data[:, 3]

all_meas = [
    {
        "t": float(t_meas[i]),
        "station": int(station_id[i]),
        "rho_m": float(rho_m[i]),
        "rho_dot_m_s": float(rho_dot_m_s[i]),
    }
    for i in range(len(t_meas))
]

# -----------------------------
# Station mapping (state order: Rs_101, Rs_337, Rs_394)
# -----------------------------
station_state_map = {101: 0, 337: 1, 394: 2}

# -----------------------------
# Constants for dynamics (Project 1 handout)
# -----------------------------
Re = 6378136.3  # meters
omega_vec = np.array([0.0, 0.0, 7.2921158553e-5], dtype=float)

const = {
    "Re": Re,
    "drag": True,
    "atmosphere_rotates": True,
    "omega_vec": omega_vec,
    "rho0": 3.614e-13,
    "r0": 700000.0 + Re,
    "H": 88667.0,
    "A": 3.0,
    "m": 970.0,
}

dyn_fun = lambda t, x: f_muJ2_drag(t, x, const)
dyn_jac = lambda t, x: A_muJ2_drag(t, x, const)

prop_settings = PropSettings(rtol=1e-10, atol=1e-10, method="DOP853")

# -----------------------------
# Station initial positions (ECEF at t0, used as ECI at t0)
# -----------------------------
Rs_101 = np.array([-5127510.0, -3794160.0, 0.0], dtype=float)
Rs_337 = np.array([3860910.0, 3238490.0, 3898094.0], dtype=float)
Rs_394 = np.array([549505.0, -1380872.0, 6182197.0], dtype=float)

# -----------------------------
# Initial guess 
# -----------------------------
r0 = np.array([757700.0, 5222607.0, 4851500.0], dtype=float)
v0 = np.array([2213.21, 4678.34, -5371.30394], dtype=float)
mu0 = 3.986004415e14
J2_0 = 1.082626925638815e-3
Cd0 = 2.2

x0_bar = np.hstack([r0, v0, mu0, J2_0, Cd0, Rs_101, Rs_337, Rs_394])

# -----------------------------
# Measurement noise
# TODO: update these sigmas if different
# -----------------------------
sigma_rho_m = 0.01      # 1 cm
sigma_rhod_m_s = 0.001  # 1 mm/s
R = np.diag([sigma_rho_m**2, sigma_rhod_m_s**2])

# -----------------------------
# A priori covariance 
# -----------------------------
P0_diag = (
    [1e6] * 6 +          # r,v
    [1e20, 1e6, 1e6] +    # mu, J2, Cd
    [1e-10] * 3 +         # station 101
    [1e6] * 6             # stations 337, 394
)
P0 = np.diag(P0_diag)

# -----------------------------
# Run batch (18-state)
# -----------------------------
x0_hat, P0_hat, info = batch_estimate_x0(
    all_meas=all_meas,
    stations=[],  # not used in 18-state mode
    x0_bar=x0_bar,
    P0=P0,
    R=R,
    dyn_fun=dyn_fun,
    dyn_jac=dyn_jac,
    station_state_map=station_state_map,
    max_iter=3,
    tol=1e-8,
    prop_settings=prop_settings,
    station_start_index=9,
    num_stations=3,
    omega_vec=omega_vec,
)

print("\n=== Batch finished (18-state) ===")
print("num_iters:", info["num_iters"])
print("x0_hat (first 6):", x0_hat[:6])
print("diag(P0_hat) (first 6):", np.diag(P0_hat)[:6])

# -----------------------------
# Post-process + RMS prints
# -----------------------------
result = run_batch_post_processing_18(
    all_meas=all_meas,
    x0_hat=x0_hat,
    P0_hat=P0_hat,
    info=info,
    truth_times=None,
    truth_states_6=None,
    length_unit_in="m",
    length_unit_out="km",
    first_pass_gap_s=6 * 3600.0,
)

print_rms_summary(result, ignore_first_pass=False)
print_rms_summary(result, ignore_first_pass=True)

# -----------------------------
# Make plots
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "Plots" / "Batch_18"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

make_prefit_residuals_plot(result, PLOT_DIR)
make_postfit_residuals_linear_plot(result, PLOT_DIR)
make_postfit_residuals_nonlinear_plot(result, PLOT_DIR)
make_cov_diag_log_plot(result, PLOT_DIR)
make_trace_cov_plot(result, PLOT_DIR)

# -----------------------------
# Per-iteration residual plots (iterations 1-3)
# -----------------------------
prefit_hist = info.get("prefit_resids_hist", [])
postfit_lin_hist = info.get("postfit_resids_linear_hist", [])

scale_m_to_km = 1e-3
max_iters_to_plot = min(3, len(prefit_hist), len(postfit_lin_hist))

for k in range(max_iters_to_plot):
    iter_dir = PLOT_DIR / f"Iter_{k+1}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    result_k = replace(
        result,
        prefit_resids_final=np.asarray(prefit_hist[k], dtype=float) * scale_m_to_km,
        postfit_resids_linear_final=np.asarray(postfit_lin_hist[k], dtype=float) * scale_m_to_km,
    )

    make_prefit_residuals_plot(result_k, iter_dir)
    make_postfit_residuals_linear_plot(result_k, iter_dir)

# State error plots require truth
if result.state_error_meas is not None:
    make_state_errors_eci_plots(result, PLOT_DIR)
    make_state_errors_rsw_plot(result, PLOT_DIR)

print(f"\nSaved plots to: {PLOT_DIR}")
