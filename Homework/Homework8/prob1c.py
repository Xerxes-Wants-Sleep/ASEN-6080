import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.integrate import solve_ivp

sys.path.append("../../")

from src.Functions.dynamics_muJ2_drag import f_muJ2_drag
from src.helpers.plotting.common import savefig
from src.helpers.plotting.plot_corner_mc import corner_plot_mc


script_dir = Path(__file__).resolve().parent
mc_dir = script_dir / "mc_data"
mc_files = sorted(mc_dir.glob("prob1a_mc_*_cases_24h.npz"), key=lambda p: p.stat().st_mtime)
mc_path = mc_files[-1]

data = np.load(mc_path)
t_eval_s = np.asarray(data["t_eval_s"], dtype=float)
state_hist_m = np.asarray(data["state_hist_eci_m_mps"], dtype=float)  # (N, Nt, 6), m and m/s
success = np.asarray(data["success"], dtype=bool)
x0_nominal_9 = np.asarray(data["x0_nominal_9"], dtype=float).reshape(9)

state_hist_m = state_hist_m[success, :, :]

epoch_hr = np.arange(0.0, 24.0 + 1e-12, 4.0, dtype=float)
epoch_s = 3600.0 * epoch_hr
idx_epochs = np.searchsorted(t_eval_s, epoch_s)
if not np.allclose(t_eval_s[idx_epochs], epoch_s):
    raise ValueError("Requested epochs are not exactly represented in t_eval_s. Check dt_out_s in prob1a.")

npz_keys = set(data.files)
Re_m = float(data["Re_m"]) if "Re_m" in npz_keys else float(data["r0_m"]) - 700000.0
const = {
    "Re": Re_m,
    "drag": True,
    "atmosphere_rotates": True,
    "omega_vec": np.array([0.0, 0.0, float(data["omega_earth_rad_s"]) if "omega_earth_rad_s" in npz_keys else 7.2921158553e-5], dtype=float),
    "rho0": float(data["rho0_kg_m3"]),
    "r0": float(data["r0_m"]),
    "H": float(data["H_m"]),
    "A": float(data["area_m2"]),
    "m": float(data["mass_kg"]),
}

sol_nom = solve_ivp(
    fun=lambda t, x: f_muJ2_drag(t, x, const),
    t_span=(float(epoch_s[0]), float(epoch_s[-1])),
    y0=x0_nominal_9,
    t_eval=epoch_s,
    rtol=1e-10,
    atol=1e-10,
    method="DOP853",
)
if not sol_nom.success:
    raise RuntimeError(f"Nominal propagation failed in prob1c: {sol_nom.message}")
nom_hist_m = sol_nom.y[0:6, :].T  # (Ne, 6)

# Convert outputs to km / km/s for plotting and reporting.
state_hist_u = state_hist_m.copy()
state_hist_u[:, :, 0:3] *= 1.0e-3
state_hist_u[:, :, 3:6] *= 1.0e-3

nom_hist_u = nom_hist_m.copy()
nom_hist_u[:, 0:3] *= 1.0e-3
nom_hist_u[:, 3:6] *= 1.0e-3

plot_dir = script_dir / "Plots" / "prob1c_corner_2Cd"
plot_dir.mkdir(parents=True, exist_ok=True)

state_labels = ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]
axis_labels = ["X [km]", "Y [km]", "Z [km]", "Xdot [km/s]", "Ydot [km/s]", "Zdot [km/s]"]

rows = []

for k, hr in enumerate(epoch_hr):
    idx = int(idx_epochs[k])
    Xk = state_hist_u[:, idx, :]  # (N, 6)
    mu_k = np.mean(Xk, axis=0)
    P_k = np.cov(Xk, rowvar=False)
    sig_k = np.sqrt(np.maximum(np.diag(P_k), 0.0))
    nom_k = nom_hist_u[k, :]

    fig = corner_plot_mc(
        prop_traj=nom_k,
        monte_traj=Xk.T,
        P=P_k,
        title_text=f"Monte Carlo Corner Plot at t = {int(hr)} hr",
        labels=axis_labels,
    )
    savefig(fig, plot_dir / f"corner_t_{int(hr):02d}h.png", tight_layout=False)

    epoch_df = pd.DataFrame(
        {
            "state": state_labels,
            "sigma": sig_k,
            "mean": mu_k,
            "nominal": nom_k,
        }
    )
    print(f"\nEpoch: {int(hr)} hr")
    with pd.option_context("display.float_format", lambda x: f"{x:.6e}"):
        print(epoch_df.to_string(index=False))

    for i, s in enumerate(state_labels):
        rows.append(
            {
                "epoch_hr": float(hr),
                "state": s,
                "sigma": float(sig_k[i]),
                "mean": float(mu_k[i]),
                "nominal": float(nom_k[i]),
            }
        )

summary_df = pd.DataFrame(rows)
summary_csv = plot_dir / "prob1c_epoch_state_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print(f"\nLoaded MC data: {mc_path}")
print(f"Used samples: {state_hist_u.shape[0]}")
print(f"Saved corner plots to: {plot_dir}")
print(f"Saved summary CSV: {summary_csv}")
