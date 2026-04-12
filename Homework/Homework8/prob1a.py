##### Monte Carlo Orbit Propagation #####

import numpy as np
import sys
from pathlib import Path
from scipy.integrate import solve_ivp

sys.path.append("../../")
from src.Functions.kep2cart import Keplarian_to_Cartesian
from src.Functions.dynamics_muJ2_drag import f_muJ2_drag

# -----------------------------
# User inputs / configuration
# -----------------------------
oe_params = {
    "a": 6750,  # semi-major axis [km]
    "e": 0.001,  # eccentricity
    "i": 30,  # inclination [deg]
    "Omega": 0.0,  # RAAN [deg]
    "omega": 0.0,  # argument of periapsis [deg]
    "f": 0.0,  # true anomaly [deg]
}

N_samples = 3000  # number of Monte Carlo cases to propagate
seed = 6080
t0_s = 0.0
tf_s = 24.0 * 3600.0
dt_out_s = 60.0

sigma_r_m = 1000.0  # 1 km
sigma_v_m_s = 1.0  # 1 m/s

mu: np.float64 = 398600.4415
Re: np.float64 = 6378.1363
J2: np.float64 = 0.0010826269
Cd: np.float64 = 2

# Drag model settings (SI units)
area_m2 = 3.0
mass_kg = 970.0
omega_earth_rad_s = 7.2921158553e-5
rho0_kg_m3 = 3.614e-13
r0_m = Re * 1e3 + 700e3
H_m = 88667.0

rtol = 1e-10
atol = 1e-10
method = "DOP853"

r_N, v_N, _ = Keplarian_to_Cartesian(
    mu,
    oe_params["a"],
    oe_params["e"],
    oe_params["i"],
    oe_params["Omega"],
    oe_params["omega"],
    oe_params["f"],
)
r_N_m = np.asarray(r_N, dtype=float) * 1e3
v_N_m_s = np.asarray(v_N, dtype=float) * 1e3
mu_m3_s2 = float(mu) * 1e9
Re_m = float(Re) * 1e3
x0_nom_9 = np.hstack((r_N_m, v_N_m_s, mu_m3_s2, float(J2), float(Cd)))


def propagate_mc_24h(x0_samples_9: np.ndarray, t_eval_s: np.ndarray, const: dict) -> tuple[np.ndarray, np.ndarray]:
    n_samples = x0_samples_9.shape[0]
    n_times = t_eval_s.size

    state_hist_6 = np.full((n_samples, n_times, 6), np.nan, dtype=float)
    success = np.zeros(n_samples, dtype=bool)

    rhs = lambda t, x: f_muJ2_drag(t, x, const)

    for k in range(n_samples):
        sol = solve_ivp(
            fun=rhs,
            t_span=(float(t_eval_s[0]), float(t_eval_s[-1])),
            y0=x0_samples_9[k, :],
            t_eval=t_eval_s,
            rtol=rtol,
            atol=atol,
            method=method,
        )

        if sol.success and sol.y.shape[1] == n_times:
            state_hist_6[k, :, :] = sol.y[0:6, :].T
            success[k] = True
        else:
            success[k] = False

        if (k + 1) % 100 == 0 or (k + 1) == n_samples:
            print(f"Propagated {k + 1}/{n_samples} cases")

    return state_hist_6, success


script_dir = Path(__file__).resolve().parent
out_dir = script_dir / "mc_data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"prob1a_mc_{N_samples}_cases_24h.npz"

t_eval_s = np.arange(t0_s, tf_s + dt_out_s, dt_out_s, dtype=float)

rng = np.random.default_rng(seed)
sig6 = np.array(
    [sigma_r_m, sigma_r_m, sigma_r_m, sigma_v_m_s, sigma_v_m_s, sigma_v_m_s],
    dtype=float,
)
dx0 = rng.normal(loc=0.0, scale=sig6, size=(N_samples, 6))
x0_samples_9 = np.repeat(np.asarray(x0_nom_9, dtype=float).reshape(1, 9), N_samples, axis=0)
x0_samples_9[:, 0:6] += dx0

const = {
    "Re": Re_m,
    "drag": True,
    "atmosphere_rotates": True,
    "omega_vec": np.array([0.0, 0.0, omega_earth_rad_s], dtype=float),
    "rho0": rho0_kg_m3,
    "r0": r0_m,
    "H": H_m,
    "A": area_m2,
    "m": mass_kg,
}

state_hist_6, success = propagate_mc_24h(x0_samples_9, t_eval_s, const)

np.savez_compressed(
    out_path,
    t_eval_s=t_eval_s,
    state_hist_eci_m_mps=state_hist_6,
    x0_samples_9=x0_samples_9,
    x0_nominal_9=x0_nom_9,
    success=success,
    seed=np.array(seed, dtype=int),
    sigma_r_m=np.array(sigma_r_m, dtype=float),
    sigma_v_m_s=np.array(sigma_v_m_s, dtype=float),
    mu_m3_s2=np.array(mu_m3_s2, dtype=float),
    J2=np.array(J2, dtype=float),
    Cd=np.array(Cd, dtype=float),
    area_m2=np.array(area_m2, dtype=float),
    mass_kg=np.array(mass_kg, dtype=float),
    rho0_kg_m3=np.array(rho0_kg_m3, dtype=float),
    r0_m=np.array(r0_m, dtype=float),
    H_m=np.array(H_m, dtype=float),
    oe_a_km=np.array(oe_params["a"], dtype=float),
    oe_e=np.array(oe_params["e"], dtype=float),
    oe_i_deg=np.array(oe_params["i"], dtype=float),
    oe_Omega_deg=np.array(oe_params["Omega"], dtype=float),
    oe_omega_deg=np.array(oe_params["omega"], dtype=float),
    oe_f_deg=np.array(oe_params["f"], dtype=float),
)

n_ok = int(np.count_nonzero(success))
print(f"\nSaved MC output: {out_path}")
print(f"Successful runs: {n_ok}/{N_samples}")
print(f"Stored history shape: {state_hist_6.shape} (samples, times, state6)")

if n_ok > 0:
    xf = state_hist_6[success, -1, :]
    mean_f = np.mean(xf, axis=0)
    std_f = np.std(xf, axis=0, ddof=1) if n_ok > 1 else np.zeros(6)
    print("Final-state mean [x y z vx vy vz] (m, m/s):")
    print(mean_f)
    print("Final-state std  [x y z vx vy vz] (m, m/s):")
    print(std_f)


