import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.stats import chi2
sys.path.append("../../")

from src.Functions.dynamics_muJ2_drag import f_muJ2_drag, A_muJ2_drag
from src.Functions.propagation import PropSettings, propagate_x_phi_history
from src.helpers.plotting.common import savefig
from src.helpers.plotting.plot_corner_mc import corner_plot_mc


script_dir = Path(__file__).resolve().parent
mc_dir = script_dir / "mc_data"
mc_files = sorted(mc_dir.glob("prob1a_mc_3000_cases_24h.npz"), key=lambda p: p.stat().st_mtime)
mc_path = mc_files[-1]

data = np.load(mc_path)
t_eval_s = np.asarray(data["t_eval_s"], dtype=float)
state_hist_m = np.asarray(data["state_hist_eci_m_mps"], dtype=float)  # (N, Nt, 6), m and m/s
success = np.asarray(data["success"], dtype=bool)
x0_nominal_9 = np.asarray(data["x0_nominal_9"], dtype=float).reshape(9)

state_hist_m = state_hist_m[success, :, :]
x0_nominal_6 = x0_nominal_9[:6]
mu = float(x0_nominal_9[6])
J2 = float(x0_nominal_9[7])
Cd = float(x0_nominal_9[8])

const = {
    "Re": float(data["r0_m"]) - 700000.0,
    "drag": True,
    "atmosphere_rotates": True,
    "omega_vec": np.array([0.0, 0.0, 7.2921158553e-5], dtype=float),
    "rho0": float(data["rho0_kg_m3"]),
    "r0": float(data["r0_m"]),
    "H": float(data["H_m"]),
    "A": float(data["area_m2"]),
    "m": float(data["mass_kg"]),
}


def f6(t: float, x6: np.ndarray) -> np.ndarray:
    x9 = np.hstack((np.asarray(x6, dtype=float).reshape(6), mu, J2, Cd))
    return f_muJ2_drag(t, x9, const)[0:6]


def A6(t: float, x6: np.ndarray) -> np.ndarray:
    x9 = np.hstack((np.asarray(x6, dtype=float).reshape(6), mu, J2, Cd))
    return A_muJ2_drag(t, x9, const)[0:6, 0:6]


settings = PropSettings(rtol=1e-10, atol=1e-10, method="DOP853")
Xhat_hist_m, Phi_i0 = propagate_x_phi_history(
    x0=x0_nominal_6,
    t_eval=t_eval_s,
    f=f6,
    A=A6,
    settings=settings,
)

sigma_r_m = float(data["sigma_r_m"])
sigma_v_m_s = float(data["sigma_v_m_s"])
P0 = np.diag(
    [
        sigma_r_m**2,
        sigma_r_m**2,
        sigma_r_m**2,
        sigma_v_m_s**2,
        sigma_v_m_s**2,
        sigma_v_m_s**2,
    ]
)
P_hist_m = np.einsum("tij,jk,tlk->til", Phi_i0, P0, Phi_i0)

epoch_hr = np.arange(0.0, 24.0 + 1e-12, 4.0, dtype=float)
epoch_s = 3600.0 * epoch_hr
idx_epochs = np.searchsorted(t_eval_s, epoch_s)
if not np.allclose(t_eval_s[idx_epochs], epoch_s):
    raise ValueError("Requested epochs are not exactly represented in t_eval_s. Check dt_out_s in prob1a.")

# Convert to km / km/s for plotting + printed tables (same style as prob1c).
state_hist_u = state_hist_m.copy()
state_hist_u[:, :, 0:3] *= 1.0e-3
state_hist_u[:, :, 3:6] *= 1.0e-3

Xhat_hist_u = Xhat_hist_m.copy()
Xhat_hist_u[:, 0:3] *= 1.0e-3
Xhat_hist_u[:, 3:6] *= 1.0e-3

P_hist_u = P_hist_m.copy()
P_hist_u[:, 0:3, :] *= 1.0e-3
P_hist_u[:, :, 0:3] *= 1.0e-3
P_hist_u[:, 3:6, :] *= 1.0e-3
P_hist_u[:, :, 3:6] *= 1.0e-3

plot_dir = script_dir / "Plots" / "prob2b_linear_cov"
plot_dir.mkdir(parents=True, exist_ok=True)

state_labels = ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]
axis_labels = ["X [km]", "Y [km]", "Z [km]", "Xdot [km/s]", "Ydot [km/s]", "Zdot [km/s]"]

rows_state = []
rows_cov = []

for k, hr in enumerate(epoch_hr):
    idx = int(idx_epochs[k])
    Xk = state_hist_u[:, idx, :]      # (N, 6)
    xhat_k = Xhat_hist_u[idx, :]       # (6,)
    Pk = P_hist_u[idx, :, :]           # (6,6)

    mu_mc = np.mean(Xk, axis=0)
    sig_lin = np.sqrt(np.maximum(np.diag(Pk), 0.0))
    err = Xk - xhat_k.reshape(1, 6)
    solve_term = np.linalg.solve(Pk, err.T)                      # (6, N)
    d2 = np.sum(err.T * solve_term, axis=0)                     # (N,)
    threshold_95 = chi2.ppf(0.95, df=6)
    inside_conf = d2 <= threshold_95
    pct_mahal = 100.0 * float(np.mean(inside_conf))

    fig = corner_plot_mc(
        prop_traj=xhat_k,
        monte_traj=Xk.T,
        P=Pk,
        title_text=f"Linear Cov Corner Plot at t = {int(hr)} hr",
        labels=axis_labels,
    )
    savefig(fig, plot_dir / f"corner_linear_t_{int(hr):02d}h.png", tight_layout=False)

    epoch_df = pd.DataFrame(
        {
            "state": state_labels,
            "sigma_linear": sig_lin,
            "mean_mc": mu_mc,
            "nominal": xhat_k,
        }
    )
    print(f"\nEpoch: {int(hr)} hr")
    with pd.option_context("display.float_format", lambda x: f"{x:.6e}"):
        print(epoch_df.to_string(index=False))
    print(f"Inside 2sigma ellipsoid via Mahalanobis [%]: {pct_mahal:.6f}")

    for i, s in enumerate(state_labels):
        rows_state.append(
            {
                "epoch_hr": float(hr),
                "state": s,
                "sigma_linear": float(sig_lin[i]),
                "mean_mc": float(mu_mc[i]),
                "nominal": float(xhat_k[i]),
            }
        )
    rows_cov.append(
        {
            "epoch_hr": float(hr),
            "state": "mahal_2sigma_ellipsoid",
            "inside_2sigma_pct": float(pct_mahal),
        }
    )

state_df = pd.DataFrame(rows_state)
cov_df = pd.DataFrame(rows_cov)

state_csv = plot_dir / "prob2b_epoch_state_summary.csv"
cov_csv = plot_dir / "prob2b_2sigma_coverage_summary.csv"
state_df.to_csv(state_csv, index=False)
cov_df.to_csv(cov_csv, index=False)

print(f"\nLoaded MC data: {mc_path}")
print(f"Used samples: {state_hist_u.shape[0]}")
print(f"Saved corner plots to: {plot_dir}")
print(f"Saved state summary CSV: {state_csv}")
print(f"Saved coverage CSV: {cov_csv}")
