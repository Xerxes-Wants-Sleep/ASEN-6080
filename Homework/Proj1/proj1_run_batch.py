import numpy as np
import sys
from dataclasses import replace

from pathlib import Path
sys.path.append("../../")

from src.Functions.batch_18_state import batch_estimate_x0
from src.Functions.dynamics_muJ2_drag import f_muJ2_drag, A_muJ2_drag
from src.Functions.propagation import PropSettings, propagate_x_phi_history

from src.helpers.plotting.post_processing import run_batch_post_processing_18, print_rms_summary
from src.helpers.plotting.plot_prefit_residuals import make_prefit_residuals_plot
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_postfit_residuals_nonlinear import make_postfit_residuals_nonlinear_plot
from src.helpers.plotting.plot_cov_diag_log import make_cov_diag_log_plot
from src.helpers.plotting.plot_trace_cov import make_trace_cov_plot
from src.helpers.plotting.plot_state_errors_eci import make_state_errors_eci_plots
from src.helpers.plotting.plot_state_errors_rsw import make_state_errors_rsw_plot
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.plot_cov_ellipsoid import plot_cov_ellipsoid
from src.helpers.plotting.plot_state_errors_18 import make_state_errors_18_plot


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
# Station mapping 
# -----------------------------
station_state_map = {101: 0, 337: 1, 394: 2}

# -----------------------------
# Constants for dynamics
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
v0 = np.array([2213.21, 4678.34, -5371.30], dtype=float)
mu0 = 3.986004415e14
J2_0 = 1.082626925638815e-3
Cd0 = 2

x0_bar = np.hstack([r0, v0, mu0, J2_0, Cd0, Rs_101, Rs_337, Rs_394])
print(x0_bar)
# -----------------------------
# Measurement noise
# -----------------------------
sigma_rho_m = .01      # 1 cm
sigma_rhod_m_s = .001  # 1 mm/s
R = np.diag([sigma_rho_m**2, sigma_rhod_m_s**2])

# -----------------------------
# A priori covariance 
# -----------------------------
P0_diag = [
    1e6,   # 1  r_x
    1e6,   # 2  r_y
    1e6,   # 3  r_z
    1e6,   # 4  v_x
    1e6,   # 5  v_y
    1e6,   # 6  v_z

    1e20,  # 7  mu
    1e6,   # 8  J2
    1e6,   # 9  Cd

    1e-10, # 10 station 101 x
    1e-10, # 11 station 101 y
    1e-10, # 12 station 101 z

    1e6,   # 13 station 394 x
    1e6,   # 14 station 394 y
    1e6,   # 15 station 337 z

    1e6,   # 16 station 394 x
    1e6,   # 17 station 394 y
    1e6    # 18 station 394 z
]

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
# Formal 1-sigma at reference epoch (from P0_hat)
# -----------------------------
sig_pos = np.sqrt(np.diag(P0_hat)[0:3])
sig_vel = np.sqrt(np.diag(P0_hat)[3:6])
print("Final 1-sigma at t0 (from P0_hat):")
print(f"  sigma_x = {sig_pos[0]:.6g} m, sigma_y = {sig_pos[1]:.6g} m, sigma_z = {sig_pos[2]:.6g} m")
print(f"  sigma_vx = {sig_vel[0]:.6g} m/s, sigma_vy = {sig_vel[1]:.6g} m/s, sigma_vz = {sig_vel[2]:.6g} m/s")
print(
    "SIGMA_ROW, "
    f"{sig_pos[0]:.6g}, {sig_pos[1]:.6g}, {sig_pos[2]:.6g}, "
    f"{sig_vel[0]:.6g}, {sig_vel[1]:.6g}, {sig_vel[2]:.6g}"
)

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
    length_unit_out="m",
    first_pass_gap_s=6 * 3600.0,
)

print_rms_summary(result, ignore_first_pass=False)

# -----------------------------
# Make plots
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "Plots" / "Brun7"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

make_prefit_residuals_plot(result, PLOT_DIR)
make_postfit_residuals_linear_plot(result, PLOT_DIR)
# make_postfit_residuals_nonlinear_plot(result, PLOT_DIR)
# make_cov_diag_log_plot(result, PLOT_DIR)
# make_trace_cov_plot(result, PLOT_DIR)
make_trace_cov_pos_vel_plot(result, PLOT_DIR, length_unit="m")
plot_cov_ellipsoid(result, PLOT_DIR)

# -----------------------------
# State error (18-state) vs a priori flow
#   Delta x = Phi(t, x0_apriori, t0) - X_hat(t)
# -----------------------------
dx_hist_time = info.get("dx_hist_time", None)
if dx_hist_time is not None:
    make_state_errors_18_plot(
        t_meas,
        -dx_hist_time,
        PLOT_DIR,
        title="Batch State Error (X_ref - X_hat)",
        filename="state_errors_18_batch.png",
    )

# -----------------------------
# Per-iteration plots (iterations 1-3)
# -----------------------------
prefit_hist = info.get("prefit_resids_hist", [])
postfit_lin_hist = info.get("postfit_resids_linear_hist", [])
Lambda_hist = info.get("Lambda_hist", [])
x0_star_hist = info.get("x0_star_hist", [])

max_iters_to_plot = min(3, len(prefit_hist), len(postfit_lin_hist), len(Lambda_hist))

# Baseline (a-priori flow) for per-iteration state error plots
X_base_iter, _ = propagate_x_phi_history(
    x0=x0_bar,
    t_eval=t_meas,
    f=dyn_fun,
    A=dyn_jac,
    settings=prop_settings,
)

for k in range(max_iters_to_plot):
    iter_dir = PLOT_DIR / f"Iter_{k+1}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Residual plots for this iteration
    result_k = replace(
        result,
        prefit_resids_final=np.asarray(prefit_hist[k], dtype=float),
        postfit_resids_linear_final=np.asarray(postfit_lin_hist[k], dtype=float),
    )
    make_prefit_residuals_plot(result_k, iter_dir)
    make_postfit_residuals_linear_plot(result_k, iter_dir)

    # Propagate this iteration's estimate to build covariance history
    if len(x0_star_hist) > (k + 1):
        x0_iter = np.asarray(x0_star_hist[k + 1], dtype=float)
        P0_iter = np.linalg.inv(np.asarray(Lambda_hist[k], dtype=float))

        X_iter, Phi_iter = propagate_x_phi_history(
            x0=x0_iter,
            t_eval=t_meas,
            f=dyn_fun,
            A=dyn_jac,
            settings=prop_settings
        )

        # State error vs baseline for this iteration
        make_state_errors_18_plot(
            t_meas,
            X_base_iter - X_iter,
            iter_dir,
            title=f"Batch State Error (Iter {k+1}: X_ref - X_hat)",
            filename=f"state_errors_18_iter_{k+1}.png",
        )

        n_iter = X_iter.shape[1]
        P_hist_iter = np.zeros((len(t_meas), n_iter, n_iter), dtype=float)
        for i in range(len(t_meas)):
            Phi_i = Phi_iter[i, :, :]
            P_hist_iter[i, :, :] = Phi_i @ P0_iter @ Phi_i.T

        X_iter_m = X_iter[:, 0:6]
        P_iter_m = P_hist_iter[:, 0:6, 0:6]

        result_cov_k = replace(result, xhat_meas=X_iter_m, P_meas=P_iter_m)
        make_trace_cov_pos_vel_plot(
            result_cov_k,
            iter_dir,
            filename="trace_cov_pos_vel.png",
            title=f"Trace of Covariance (pos/vel) (Iter {k+1})",
            length_unit="m",
        )
        plot_cov_ellipsoid(
            result_cov_k,
            iter_dir,
            filename="final_covariance_ellipsoids_stats.png",
            title_prefix=f"Iter {k+1}"
        )
