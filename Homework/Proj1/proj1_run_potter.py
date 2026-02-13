import numpy as np
import sys

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / "Functions"))
sys.path.append(str(SCRIPT_DIR / "plotting"))
sys.path.append("../../")

from potter18 import PotterLKF18State, potter_post_process_18
from dynamics_muJ2_drag import f_muJ2_drag, A_muJ2_drag
from src.Functions.propagation import PropSettings, propagate_x_phi_history

from src.helpers.plotting.post_processing import print_rms_summary
from src.helpers.plotting.plot_prefit_residuals import make_prefit_residuals_plot
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_postfit_residuals_nonlinear import make_postfit_residuals_nonlinear_plot
from src.helpers.plotting.plot_cov_diag_log import make_cov_diag_log_plot
from src.helpers.plotting.plot_trace_cov import make_trace_cov_plot
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.plot_cov_ellipsoid import plot_cov_ellipsoid
from plot_state_errors_18 import make_state_errors_18_plot


# -----------------------------
# Load measurements
# -----------------------------
meas_path = SCRIPT_DIR / "Given_Data" / "project.txt"
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

X0_star = np.hstack([r0, v0, mu0, J2_0, Cd0, Rs_101, Rs_337, Rs_394])

# -----------------------------
# Measurement noise
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
# Process noise (set to zero by default)
# -----------------------------
Q = np.zeros((18, 18), dtype=float)

# -----------------------------
# Run Potter LKF (18-state)
# -----------------------------
lkf = PotterLKF18State(
    X0_star=X0_star,
    P0=P0,
    R=R,
    Q=Q,
    dyn_fun=dyn_fun,
    dyn_jac=dyn_jac,
    station_state_map=station_state_map,
    prop_settings=prop_settings,
    station_start_index=9,
    num_stations=3,
    omega_vec=omega_vec,
)

out = lkf.run(all_meas, stations=None, Xtrue_meas=None)

result = potter_post_process_18(out=out, length_unit_in="m", length_unit_out="m")
print_rms_summary(result, ignore_first_pass=False)
print_rms_summary(result, ignore_first_pass=True)

# -----------------------------
# Make plots
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "Plots" / "Potter"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

make_prefit_residuals_plot(result, PLOT_DIR)
make_postfit_residuals_linear_plot(result, PLOT_DIR)
make_postfit_residuals_nonlinear_plot(result, PLOT_DIR)
make_cov_diag_log_plot(result, PLOT_DIR)
make_trace_cov_plot(result, PLOT_DIR)
make_trace_cov_pos_vel_plot(result, PLOT_DIR, length_unit="m")
plot_cov_ellipsoid(result, PLOT_DIR)

# -----------------------------
# State error (18-state) vs a priori flow
# -----------------------------
X_base, _ = propagate_x_phi_history(
    x0=X0_star,
    t_eval=out["t_meas"],
    f=dyn_fun,
    A=dyn_jac,
    settings=prop_settings,
)
state_err_18 = X_base - out["X_pf"]
make_state_errors_18_plot(
    out["t_meas"],
    state_err_18,
    PLOT_DIR,
    title="Potter LKF State Error (X_ref - X_hat)",
    filename="state_errors_18_potter.png",
)

print(f"\nSaved plots to: {PLOT_DIR}")
