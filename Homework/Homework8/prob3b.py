import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.stats import chi2

sys.path.append("../../")

from src.Functions.dynamics_muJ2_drag import f_muJ2_drag
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


def f6(t: float, x6: np.ndarray) -> np.ndarray:
    x9 = np.hstack((np.asarray(x6, dtype=float).reshape(6), mu, J2, Cd))
    return f_muJ2_drag(t, x9, const)[0:6]


def symm(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)


def chol_spd(P: np.ndarray) -> np.ndarray:
    Ps = symm(np.asarray(P, dtype=float))
    try:
        return np.linalg.cholesky(Ps)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(Ps)
        floor = max(1e-18, 1e-14 * max(float(np.max(np.abs(evals))), 1.0))
        Pspd = (evecs * np.maximum(evals, floor)) @ evecs.T
        return np.linalg.cholesky(symm(Pspd))


def project_spd(P: np.ndarray) -> np.ndarray:
    L = chol_spd(P)
    return symm(L @ L.T)


def sigma_points(x: np.ndarray, P: np.ndarray, gamma: float) -> np.ndarray:
    n = x.size
    S = chol_spd(P)
    Chi = np.zeros((n, 2 * n + 1), dtype=float)
    Chi[:, 0] = x
    for i in range(n):
        c = gamma * S[:, i]
        Chi[:, i + 1] = x + c
        Chi[:, i + 1 + n] = x - c
    return Chi


def propagate_sigma_points_step(Chi_in: np.ndarray, t0: float, t1: float) -> np.ndarray:
    n, nsig = Chi_in.shape
    y0 = Chi_in.reshape(-1, order="F")

    def rhs(tt, yflat):
        X = yflat.reshape(n, nsig, order="F")
        dX = np.zeros_like(X)
        for k in range(nsig):
            dX[:, k] = f6(tt, X[:, k])
        return dX.reshape(-1, order="F")

    sol = solve_ivp(
        rhs,
        (float(t0), float(t1)),
        y0,
        t_eval=[float(t1)],
        rtol=1e-10,
        atol=1e-10,
        method="DOP853",
    )
    return sol.y[:, -1].reshape(n, nsig, order="F")


n = 6
alpha = .8
beta = 2.0
kappa = 3.0 - n
lam = alpha**2 * (n + kappa) - n
gamma = np.sqrt(n + lam)

Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
Wm[0] = lam / (n + lam)
Wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)

Nt = t_eval_s.size
Xhat_hist_m = np.zeros((Nt, n), dtype=float)
P_hist_m = np.zeros((Nt, n, n), dtype=float)
Xhat_hist_m[0, :] = x0_nominal_6
P_hist_m[0, :, :] = P0

for k in range(1, Nt):
    x_prev = Xhat_hist_m[k - 1, :]
    P_prev = P_hist_m[k - 1, :, :]
    Chi = sigma_points(x_prev, P_prev, gamma)
    Chi_prop = propagate_sigma_points_step(Chi, float(t_eval_s[k - 1]), float(t_eval_s[k]))

    x_pred = Chi_prop @ Wm
    P_pred = np.zeros((n, n), dtype=float)
    for i in range(2 * n + 1):
        dx = (Chi_prop[:, i] - x_pred).reshape(-1, 1)
        P_pred += Wc[i] * (dx @ dx.T)
    P_pred = project_spd(P_pred)

    Xhat_hist_m[k, :] = x_pred
    P_hist_m[k, :, :] = P_pred

    if (k % 200) == 0 or (k == Nt - 1):
        print(f"UKF time update step {k}/{Nt - 1}")

epoch_hr = np.arange(0.0, 24.0 + 1e-12, 4.0, dtype=float)
epoch_s = 3600.0 * epoch_hr
idx_epochs = np.searchsorted(t_eval_s, epoch_s)

# Convert to km / km/s for plotting + printed tables (same style as prob1c/prob2b).
state_hist_u = state_hist_m.copy()
state_hist_u[:, :, 0:3] *= 1.0e-3
state_hist_u[:, :, 3:6] *= 1.0e-3

Xhat_hist_u = Xhat_hist_m.copy()
Xhat_hist_u[:, 0:3] *= 1.0e-3
Xhat_hist_u[:, 3:6] *= 1.0e-3

S = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
P_hist_u = np.einsum("ij,tjk,kl->til", S, P_hist_m, S)

plot_dir = script_dir / "Plots" / "prob3b_ukf_cov"
plot_dir.mkdir(parents=True, exist_ok=True)

state_labels = ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]
axis_labels = ["X [km]", "Y [km]", "Z [km]", "Xdot [km/s]", "Ydot [km/s]", "Zdot [km/s]"]

rows_state = []
rows_cov = []

for k, hr in enumerate(epoch_hr):
    idx = int(idx_epochs[k])
    Xk = state_hist_u[:, idx, :]      # (N, 6)
    xhat_k = Xhat_hist_u[idx, :]      # (6,)
    Pk = P_hist_u[idx, :, :]          # (6,6)

    mu_mc = np.mean(Xk, axis=0)
    sig_ukf = np.sqrt(np.maximum(np.diag(Pk), 0.0))
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
        title_text=f"UKF Cov Corner Plot at t = {int(hr)} hr",
        labels=axis_labels,
    )
    savefig(fig, plot_dir / f"corner_ukf_t_{int(hr):02d}h.png", tight_layout=False)

    epoch_df = pd.DataFrame(
        {
            "state": state_labels,
            "sigma_ukf": sig_ukf,
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
                "sigma_ukf": float(sig_ukf[i]),
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

state_csv = plot_dir / "prob3b_epoch_state_summary.csv"
cov_csv = plot_dir / "prob3b_2sigma_coverage_summary.csv"
state_df.to_csv(state_csv, index=False)
cov_df.to_csv(cov_csv, index=False)

print(f"\nLoaded MC data: {mc_path}")
print(f"Used samples: {state_hist_u.shape[0]}")
print(f"Saved corner plots to: {plot_dir}")
print(f"Saved state summary CSV: {state_csv}")
print(f"Saved coverage CSV: {cov_csv}")
