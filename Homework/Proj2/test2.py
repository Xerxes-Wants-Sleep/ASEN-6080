import numpy as np
import pandas as pd
import sys
import scipy.io
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path

sys.path.append("../../")

from src.Functions.filters import UnscentedKalmanFilter, LinearizedKalmanFilter, ExtendedKalmanFilter
from src.Functions.batch import batch_estimate_x0
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_stm_deriv, mu_sun_srp_state_deriv
from src.Functions.jacobians import cannonball_SRP, srp_thirdbody_variational_eq
from src.Functions.Ephem import ephem
from src.Functions.calcB_plane import calc_bplane
from src.Functions.SOIcheck import SOIcheck
from src.helpers.plotting.plot_bplane import make_bplane_plot

def build_stations():
    theta0_deg = 0
    w_earth_rad_per_s = 7.29211585275553e-5
    radius_earth_km = 6378.1363

    return [
        Stations(
            "DSS 34",
            lat_deg=-35.398333,
            lon_deg=148.981944,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + .691750,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "DSS 65",
            lat_deg=40.427222,
            lon_deg=-355.749444,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + .834539,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "DSS 13",
            lat_deg=35.247164,
            lon_deg=243.205000,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + 1.07114904,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
    ]

truth_traj = scipy.io.loadmat('Given_data/Project2_Prob2_truth_traj_50days.mat')
t = truth_traj["Tt_50"].squeeze()   # (2681,)
Xt = truth_traj["Xt_50"]            # (2681, 56)


# -----------------------------
# 50-day propagation on truth time grid (state+STM, 56 states total)
# -----------------------------

Jd0 = 2456296.25
mu_sun = 132712440017.987 # km^3/s^2
AU_km = 149597870.7 # km
solar_flux_W_m2 = 1357 # W/m^2 at 1 AU
SRP_area_mass_ratio = .01 # m^2/kg

# Use the same initial state as the provided truth trajectory.
state0 = np.array([-274096790.0, -92859240.0, -40199490.0, 32.67,  -8.94, -3.88, 1.2], dtype=float)


# Earth/Sun ephemeris sampled on the same time grid as truth
N = t.size
earth_r = np.zeros((N, 3), dtype=float)
earth_v = np.zeros((N, 3), dtype=float)
sun_r = np.zeros((N, 3), dtype=float)
sun_v = np.zeros((N, 3), dtype=float)

ephem_mode = "geocentric"

for k in range(N):
    jd = Jd0 + t[k] / 86400.0
    rE_km, vE_km_s, _ = ephem(jd, 3, frame="EME2000")
    rE_km = np.asarray(rE_km, dtype=float).reshape(3,)
    vE_km_s = np.asarray(vE_km_s, dtype=float).reshape(3,)

    earth_r[k, :] = 0.0
    earth_v[k, :] = 0.0
    sun_r[k, :] = -rE_km
    sun_v[k, :] = -vE_km_s

# Full initial 56-state = [state7, Phi(7x7).flatten()]
X0_56 = np.zeros(56, dtype=float)
X0_56[0:7] = state0
X0_56[7:56] = np.eye(7, dtype=float).reshape(-1)

# lightweight constants containers (no filter/prior needed)
pConst = type("pConst", (), {})()
pConst.mu_earth = 398600.4415  # km^3/s^2
pConst.mu_sun = mu_sun         # km^3/s^2

scConst = type("scConst", (), {})()
scConst.area = SRP_area_mass_ratio  # with mass=1, this gives desired A/m ratio
scConst.mass = 1.0
scConst.solar_flux_1au = solar_flux_W_m2
scConst.c = 299792458.0
scConst.AU_m = AU_km * 1000.0

def earth_state_func(tau):
    # Geocentric model: Earth is the origin in this project setup.
    return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

def sun_state_func(tau):
    # Query ephemeris at the integrator's current time
    jd = Jd0 + tau / 86400.0
    rE_km, vE_km_s, _ = ephem(jd, 3, frame="EME2000")
    rE_km = np.asarray(rE_km, dtype=float).reshape(3,)
    vE_km_s = np.asarray(vE_km_s, dtype=float).reshape(3,)
    return -rE_km, -vE_km_s

rhs = lambda tau, x: mu_sun_srp_stm_deriv(
    t=tau,
    XPhi=x,
    pConst=pConst,
    scConst=scConst,
    earth_state_func=earth_state_func,
    sun_state_func=sun_state_func,
)

sol = solve_ivp(
    fun=rhs,
    t_span=(float(t[0]), float(t[-1])),
    y0=X0_56,
    t_eval=t,
    rtol=1e-10,
    atol=1e-10,
    method="RK45",
)
if not sol.success:
    raise RuntimeError(f"Propagation failed: {sol.message}")

Xprop_56 = sol.y.T

# Save everything needed for dynamics-model verification
dynamics_test = {
    "t_s": t,
    "truth_56": Xt,
    "X0_56": X0_56,
    "Xprop_56": Xprop_56,
    "earth_r_km": earth_r,
    "earth_v_km_s": earth_v,
    "sun_r_km": sun_r,
    "sun_v_km_s": sun_v,
    "state_err_7": Xprop_56[:, 0:7] - Xt[:, 0:7],
    "stm_err_49": Xprop_56[:, 7:56] - Xt[:, 7:56],
}

print("Propagation complete.")
print("Saved keys:", list(dynamics_test.keys()))

# Final-time STM difference (7x7), printed in a readable table
phi_prop_final = Xprop_56[-1, 7:56].reshape(7, 7)
phi_truth_final = Xt[-1, 7:56].reshape(7, 7, order="F")
phi_diff_final = phi_prop_final - phi_truth_final

state_labels = ["x", "y", "z", "vx", "vy", "vz", "Cr"]
phi_diff_df = pd.DataFrame(phi_diff_final, index=state_labels, columns=state_labels)

print("\nFinal-time STM difference (Phi_prop - Phi_truth):")
with pd.option_context("display.float_format", "{:.6e}".format):
    print(phi_diff_df.to_string())







# Plot state error over time (propagated - truth)
t_days = dynamics_test["t_s"] / 86400.0
err7 = dynamics_test["state_err_7"]

labels = [
    "dX [km]",
    "dY [km]",
    "dZ [km]",
    "dVx [km/s]",
    "dVy [km/s]",
    "dVz [km/s]",
    "dCr [-]",
]

fig, axs = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
for i in range(7):
    axs[i].plot(t_days, err7[:, i], linewidth=1.0)
    axs[i].set_ylabel(labels[i])
    axs[i].grid(True, alpha=0.3)

axs[-1].set_xlabel("Time [Days]")
fig.suptitle("State Error Vs Truth (50 Days)")
fig.tight_layout()

plot_dir = Path(__file__).resolve().parent / "Validation Plots"
plot_dir.mkdir(parents=True, exist_ok=True)
plot_path = plot_dir / "state_error_vs_truth_50days.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved plot: {plot_path}")



# b-plane verification

BdotR_ref_km = 14970.824
BdotT_ref_km = 9796.737

x0_3soi_search = Xt[-1, 0:7].astype(float).copy()
t0_3soi_search = float(t[-1])
tf_3soi_search = t0_3soi_search + 400.0 * 86400.0

dyn_7 = lambda tau, x: mu_sun_srp_state_deriv(
    t=tau,
    X=x,
    pConst=pConst,
    scConst=scConst,
    earth_state_func=earth_state_func,
    sun_state_func=sun_state_func,
)

soi_event = lambda tau, x: SOIcheck(tau, x)
soi_event.terminal = True
soi_event.direction = -1.0

sol_3soi = solve_ivp(
    fun=dyn_7,
    t_span=(t0_3soi_search, tf_3soi_search),
    y0=x0_3soi_search,
    events=soi_event,
    rtol=1.0e-10,
    atol=1.0e-10,
    method="RK45",
    max_step=3600.0,
)

t_3soi = float(sol_3soi.t_events[0][0])
X_3soi = np.asarray(sol_3soi.y_events[0][0], dtype=float).reshape(7)
r_target_km = 3.0 * 925000.0

P_3soi_test = np.eye(7, dtype=float)
(
    BdotR_km,
    BdotT_km,
    sig_R_km,
    sig_T_km,
    sig_RT_km2,
    X_crossing,
    P_Bplane,
    STR2ECI,
    XPhi_BPlane,
    t_BPlane,
) = calc_bplane(
    XPhi_3SOI=X_3soi,
    t_3SOI=t_3soi,
    P_3SOI=P_3soi_test,
    pConst=pConst,
    scConst=scConst,
    earth_state_func=earth_state_func,
    sun_state_func=sun_state_func,
)

bplane_summary = {
    "t_3soi_days": t_3soi / 86400.0,
    "r_target_3soi_km": r_target_km,
    "BdotR_km": BdotR_km,
    "BdotT_km": BdotT_km,
    "BdotR_ref_km": BdotR_ref_km,
    "BdotT_ref_km": BdotT_ref_km,
    "BdotR_error_km": BdotR_km - BdotR_ref_km,
    "BdotT_error_km": BdotT_km - BdotT_ref_km,
}

print("\nB-Plane Verification At 3*R_SOI:")
print(f"  crossing time: {bplane_summary['t_3soi_days']:.6f} days since epoch")
print(f"  target radius: {bplane_summary['r_target_3soi_km']:.6f} km")
print(f"  BdotR [km]: {BdotR_km:.6f} (ref {BdotR_ref_km:.6f}, err {bplane_summary['BdotR_error_km']:.6f})")
print(f"  BdotT [km]: {BdotT_km:.6f} (ref {BdotT_ref_km:.6f}, err {bplane_summary['BdotT_error_km']:.6f})")

make_bplane_plot(
    BdotR_km=BdotR_km,
    BdotT_km=BdotT_km,
    P_bplane=P_Bplane,
    outdir=plot_dir,
    filename="bplane_validation_truth_3soi.png",
    title="B-Plane Validation At 3*R_SOI",
)

print(f"Saved B-plane validation plot: {plot_dir / 'bplane_validation_truth_3soi.png'}")


# Final Test Stuff

# 2a observations in UKF-ready long format:

def load_project2_obs_for_ukf(obs_path: Path) -> list[dict]:
    """
    Read Project 2 observation file and convert to UKF measurement format:
      {"station": str, "t": float, "rho_km": float, "rho_dot_km_s": float}
    """
    # Project2a_Obs.txt has a trailing comma per row; force the first 7 real columns.
    # Without this, pandas may shift columns and misread the time field.
    df = pd.read_csv(
        obs_path,
        skipinitialspace=True,
        index_col=False,
        usecols=range(7),
    )
    df.columns = [c.strip() for c in df.columns]

    col_time = "Time since Epoch"
    station_cols = {
        "DSS 34": ("DSS34 Range (km)", "DSS34 Range-Rate (km/sec)"),
        "DSS 65": ("DSS65 Range (km)", "DSS65 Range-Rate (km/sec)"),
        "DSS 13": ("DSS13 Range (km)", "DSS13 Range-Rate (km/sec)"),
    }

    pieces = []
    for st_name, (rho_col, rhod_col) in station_cols.items():
        part = df[[col_time, rho_col, rhod_col]].rename(
            columns={
                col_time: "t",
                rho_col: "rho_km",
                rhod_col: "rho_dot_km_s",
            }
        )
        part["station"] = st_name
        pieces.append(part)

    meas = pd.concat(pieces, ignore_index=True)
    meas["t"] = pd.to_numeric(meas["t"], errors="coerce")
    meas["rho_km"] = pd.to_numeric(meas["rho_km"], errors="coerce")
    meas["rho_dot_km_s"] = pd.to_numeric(meas["rho_dot_km_s"], errors="coerce")
    meas = meas.dropna(subset=["rho_km", "rho_dot_km_s"])
    meas = meas.sort_values("t")[["station", "t", "rho_km", "rho_dot_km_s"]]

    return meas.to_dict(orient="records")


def to_plotting_result_6state_from_ukf_km(out: dict, *, R_km: np.ndarray):
    """
    Convert UKF output in km/km-s units to the shared plotting result format in m/m-s
    for 6-state plotting helpers (x,y,z,vx,vy,vz).
    """
    from src.helpers.plotting.post_processing import run_filter_post_processing

    xhat_key = "xhat_meas" if "xhat_meas" in out else "Xhat_meas"
    phat_key = "P_meas" if "P_meas" in out else "Phat_meas"

    xhat = np.asarray(out[xhat_key], dtype=float)
    P = np.asarray(out[phat_key], dtype=float)

    xhat6_m = xhat[:, 0:6] * 1000.0
    P6_m = P[:, 0:6, 0:6] * (1000.0 ** 2)
    two_sigma6_m = 2.0 * np.sqrt(np.maximum(np.diagonal(P6_m, axis1=1, axis2=2), 0.0))

    d = {
        "t_meas": np.asarray(out["t_meas"], dtype=float),
        "station_meas": list(out["station_meas"]),
        "xhat_meas": xhat6_m,
        "Xhat_meas": xhat6_m,
        "P_meas": P6_m,
        "two_sigma_meas": two_sigma6_m,
        "state_error_meas": None if out.get("state_error_meas", None) is None else np.asarray(out["state_error_meas"], dtype=float)[:, 0:6] * 1000.0,
        "prefit_resids_final": None if out.get("prefit_resids_final", None) is None else np.asarray(out["prefit_resids_final"], dtype=float) * 1000.0,
        "postfit_resids_linear_final": None if out.get("postfit_resids_linear_final", None) is None else np.asarray(out["postfit_resids_linear_final"], dtype=float) * 1000.0,
        "postfit_resids_meas": None if out.get("postfit_resids_meas", None) is None else np.asarray(out["postfit_resids_meas"], dtype=float) * 1000.0,
        "rms_final": out.get("rms_final", {}),
        "rms_by_iter": out.get("rms_by_iter", None),
        "R": np.asarray(R_km, dtype=float) * (1000.0 ** 2),
    }
    return run_filter_post_processing(out=d)


def make_all_state_3sigma_envelope_plot(out: dict, outdir: Path):
    """
    Plot estimated states with ±3σ covariance envelopes for all 7 estimated states.
    Units are native UKF units (km, km/s, unitless Cr).
    """
    t_hr = np.asarray(out["t_meas"], dtype=float) / 3600.0
    xhat_key = "xhat_meas" if "xhat_meas" in out else "Xhat_meas"
    phat_key = "P_meas" if "P_meas" in out else "Phat_meas"

    xhat = np.asarray(out[xhat_key], dtype=float)
    P = np.asarray(out[phat_key], dtype=float)
    sig3 = 3.0 * np.sqrt(np.maximum(np.diagonal(P, axis1=1, axis2=2), 0.0))

    labels = [
        "X [km]",
        "Y [km]",
        "Z [km]",
        "Vx [km/s]",
        "Vy [km/s]",
        "Vz [km/s]",
        "Cr [-]",
    ]

    fig, axs = plt.subplots(7, 1, figsize=(10, 13), sharex=True)
    for i in range(7):
        axs[i].plot(t_hr, xhat[:, i], "k", linewidth=1.0, label="Estimate")
        axs[i].plot(t_hr, xhat[:, i] + sig3[:, i], "r--", linewidth=1.0, label="+3 Sigma")
        axs[i].plot(t_hr, xhat[:, i] - sig3[:, i], "r--", linewidth=1.0, label="-3 Sigma")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc="best", fontsize=8)

    axs[-1].set_xlabel("Time [Hours]")
    fig.suptitle("Estimated States With ±3 Sigma Covariance Envelopes")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "state_estimates_all_3sigma_envelopes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_cr_3sigma_plot(out: dict, outdir: Path):
    """
    Dedicated Cr estimate with ±3σ envelope.
    """
    t_hr = np.asarray(out["t_meas"], dtype=float) / 3600.0
    xhat_key = "xhat_meas" if "xhat_meas" in out else "Xhat_meas"
    phat_key = "P_meas" if "P_meas" in out else "Phat_meas"

    xhat = np.asarray(out[xhat_key], dtype=float)
    P = np.asarray(out[phat_key], dtype=float)
    sig3_cr = 3.0 * np.sqrt(np.maximum(P[:, 6, 6], 0.0))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_hr, xhat[:, 6], "k", linewidth=1.2, label="Cr Estimate")
    ax.plot(t_hr, xhat[:, 6] + sig3_cr, "r--", linewidth=1.0, label="+3 Sigma")
    ax.plot(t_hr, xhat[:, 6] - sig3_cr, "r--", linewidth=1.0, label="-3 Sigma")
    ax.set_xlabel("Time [Hours]")
    ax.set_ylabel("Cr [-]")
    ax.set_title("Cr Estimate With ±3 Sigma Covariance Envelope")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    # Avoid axis offset text (e.g., "1e-6 + 1.00004"), which can look like Cr ~ 5.
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "cr_estimate_3sigma_envelope.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def propagate_snapshot_to_3soi_with_stm(
    *,
    x_arc: np.ndarray,
    P_arc: np.ndarray,
    t_arc: float,
    t_search_days: float = 400.0,
):
    """
    Propagate a 7-state estimate/covariance from arc-time to 3*RSOI crossing.
    Returns (t_3soi, x_3soi, P_3soi).
    """
    y0 = np.hstack((np.asarray(x_arc, dtype=float).reshape(7), np.eye(7).reshape(-1)))

    soi_event = lambda tau, y: SOIcheck(tau, y[:7])
    soi_event.terminal = True
    soi_event.direction = -1.0

    sol = solve_ivp(
        fun=lambda tau, y: mu_sun_srp_stm_deriv(
            t=tau,
            XPhi=y,
            pConst=pConst,
            scConst=scConst,
            earth_state_func=earth_state_func,
            sun_state_func=sun_state_func,
        ),
        t_span=(float(t_arc), float(t_arc) + float(t_search_days) * 86400.0),
        y0=y0,
        events=soi_event,
        rtol=1.0e-10,
        atol=1.0e-10,
        method="RK45",
        max_step=3600.0,
    )

    if (not sol.success) or (len(sol.t_events[0]) == 0):
        raise RuntimeError(f"3*RSOI crossing not found from arc at t={t_arc:.3f} s.")

    t_3soi = float(sol.t_events[0][0])
    y_3soi = np.asarray(sol.y_events[0][0], dtype=float)
    x_3soi = y_3soi[:7]
    Phi_arc_to_3soi = y_3soi[7:].reshape(7, 7)
    P_3soi = Phi_arc_to_3soi @ np.asarray(P_arc, dtype=float) @ Phi_arc_to_3soi.T

    return t_3soi, x_3soi, P_3soi


def _ellipse_points_from_cov(cov2: np.ndarray, n_sigma: float = 3.0, n_pts: int = 500):
    cov = np.asarray(cov2, dtype=float).reshape(2, 2)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    radii = float(n_sigma) * np.sqrt(vals)

    th = np.linspace(0.0, 2.0 * np.pi, int(n_pts))
    circ = np.vstack((np.cos(th), np.sin(th)))
    xy = vecs @ np.diag(radii) @ circ
    return xy[0, :], xy[1, :]


def make_bplane_arc_overlay_plot(records: list[dict], outdir: Path):
    """
    Plot B-plane estimate and 3-sigma covariance ellipse for each arc.
    Axis convention matches assignment-style figure: x=T, y=R.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    colors = ["r", "g", "b", "k"]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("T [km]")
    ax.set_ylabel("R [km]")
    ax.set_title("B-Plane Target Estimate and 3 Sigma Uncertainty by Arc Length")

    for i, rec in enumerate(records):
        col = colors[i % len(colors)]
        cov_tr = np.asarray(rec["cov_tr"], dtype=float)
        ex, ey = _ellipse_points_from_cov(cov_tr, n_sigma=3.0, n_pts=500)  # x=T, y=R

        Bt = float(rec["BdotT_km"])
        Br = float(rec["BdotR_km"])
        day = float(rec["day_selected"])

        ax.plot(
            Bt + ex,
            Br + ey,
            "-",
            color=col,
            linewidth=1.5,
            label=f"3 Sigma Ellipse, Arc {i+1} (~{day:.1f} Days)",
        )
        ax.plot(
            Bt,
            Br,
            "x",
            color=col,
            markersize=8,
            linewidth=2.0,
            label=f"Arc {i+1}: BdotT={Bt:.1f}, BdotR={Br:.1f} km",
        )

    BdotT_true = 9796.737
    BdotR_true = 14970.824
    ax.plot(
        BdotT_true,
        BdotR_true,
        "p",
        markersize=12,
        markerfacecolor="y",
        markeredgecolor="k",
        label=f"True Target: BdotT={BdotT_true:.3f}, BdotR={BdotR_true:.3f} km",
    )

    # Keep legend outside so it does not cover narrow covariance contours.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(outdir / "bplane_3sigma_by_arc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

obs_2a_path = Path(__file__).resolve().parent / "Given_data" / "Project2a_Obs.txt"
all_meas_2a = load_project2_obs_for_ukf(obs_2a_path)

# Station list matching Project 2 naming convention.
stations_2a = build_stations()

# Measurement covariance for [rho, rhodot] in km and km/s.
sigma_rho_km = 5.0e-3
sigma_rhod_km_s = .5e-6
R_batch = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

# Initial batch state/covariance scaffold for 7-state [r, v, Cr].
x0_batch = np.array([-274096790.0, -92859240.0, -40199490.0, 32.67, -8.94, -3.88, 1.2], dtype=float)
P0_batch = np.diag(
    [
        100.0**2,     # x [km]
        100.0**2,     # y [km]
        100.0**2,     # z [km]
        .1**2,        # vx [km/s]
        .1**2,        # vy [km/s]
        .1**2,        # vz [km/s]
        .1**2,        # Cr [-]
    ],
)

dyn_batch = lambda tau, x: mu_sun_srp_state_deriv(
    t=tau,
    X=x,
    pConst=pConst,
    scConst=scConst,
    earth_state_func=earth_state_func,
    sun_state_func=sun_state_func,
)

def jac_batch(tau, x):
    r_earth_km, _ = earth_state_func(tau)
    r_sun_km, _ = sun_state_func(tau)
    return srp_thirdbody_variational_eq(
        r_sc=np.asarray(x[:3], dtype=float),
        r_earth=np.asarray(r_earth_km, dtype=float),
        r_sun=np.asarray(r_sun_km, dtype=float),
        Cr=float(x[6]),
        area=float(scConst.area),
        mass=float(scConst.mass),
        mu_earth=float(pConst.mu_earth),
        mu_i=float(pConst.mu_sun),
        solar_flux_1au=float(scConst.solar_flux_1au),
        c=float(scConst.c),
        AU_m=float(scConst.AU_m),
    )

x0_batch_hat, P0_batch_hat, info_batch_2a = batch_estimate_x0(
    all_meas=all_meas_2a,
    stations=stations_2a,
    x0_bar=x0_batch,
    P0=P0_batch,
    R=R_batch,
    mu=pConst.mu_earth,
    J2=0.0,
    J3=0.0,
    Re=6378.1363,
    max_iter=8,
    tol=1.0e-8,
    reltol=1.0e-10,
    abstol=1.0e-10,
    method="RK45",
    dyn_fun=dyn_batch,
    dyn_jac=jac_batch,
)

out_batch_2a = {
    "t_meas": np.asarray(info_batch_2a["t_meas"], dtype=float),
    "station_meas": list(info_batch_2a["station_meas"]),
    "xhat_meas": np.asarray(info_batch_2a["state_hist"], dtype=float),
    "P_meas": np.asarray(info_batch_2a["P_hist"], dtype=float),
    "prefit_resids_final": np.asarray(info_batch_2a["prefit_residuals"], dtype=float),
    "postfit_resids_linear_final": np.asarray(info_batch_2a["postfit_residuals"], dtype=float),
    "postfit_resids_meas": np.asarray(info_batch_2a["postfit_residuals"], dtype=float),
    "state_error_meas": None,
    "rms_final": {},
    "rms_by_iter": None,
}
print(f"Iterated Batch 2a Run Complete. Number Of Updates: {len(out_batch_2a['t_meas'])}")
print(f"Batch Cr Estimate At Epoch: {float(x0_batch_hat[6]):.9f}")

from src.helpers.plotting.plot_prefit_residuals import make_prefit_residuals_plot
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot

final_plot_dir = Path(__file__).resolve().parent / "Validation Plots" / "Iterated Batch"
final_plot_dir.mkdir(parents=True, exist_ok=True)

result_2a = to_plotting_result_6state_from_ukf_km(out_batch_2a, R_km=R_batch)
make_prefit_residuals_plot(result_2a, final_plot_dir)
make_postfit_residuals_linear_plot(result_2a, final_plot_dir)

make_all_state_3sigma_envelope_plot(out_batch_2a, final_plot_dir)
make_cr_3sigma_plot(out_batch_2a, final_plot_dir)

# Part (h): B-Plane 3-Sigma Ellipses At 50/100/150/200 Day Arcs
arc_days_target = [50.0, 100.0, 150.0, 200.0]
t_days_batch = np.asarray(out_batch_2a["t_meas"], dtype=float) / 86400.0
xhat_batch = np.asarray(out_batch_2a["xhat_meas"], dtype=float)
P_batch = np.asarray(out_batch_2a["P_meas"], dtype=float)

bplane_records = []
for d_target in arc_days_target:
    idx = int(np.argmin(np.abs(t_days_batch - float(d_target))))
    x_arc = xhat_batch[idx, :]
    P_arc = P_batch[idx, :, :]
    t_arc = float(out_batch_2a["t_meas"][idx])
    d_sel = float(t_days_batch[idx])

    t_3soi_arc, x_3soi_arc, P_3soi_arc = propagate_snapshot_to_3soi_with_stm(
        x_arc=x_arc,
        P_arc=P_arc,
        t_arc=t_arc,
        t_search_days=400.0,
    )

    (
        BdotR_arc,
        BdotT_arc,
        _sig_R,
        _sig_T,
        _sig_RT,
        _X_cross,
        P_Bplane_arc,
        _STR2ECI,
        _XPhi_BPlane,
        _t_BPlane,
    ) = calc_bplane(
        XPhi_3SOI=x_3soi_arc,
        t_3SOI=t_3soi_arc,
        P_3SOI=P_3soi_arc,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    # [T,R] covariance for assignment axis convention x=T, y=R
    cov_tr = np.array(
        [
            [P_Bplane_arc[1, 1], P_Bplane_arc[1, 2]],
            [P_Bplane_arc[1, 2], P_Bplane_arc[2, 2]],
        ],
        dtype=float,
    )

    bplane_records.append(
        {
            "day_target": float(d_target),
            "day_selected": d_sel,
            "idx": idx,
            "t_arc_s": t_arc,
            "t_3soi_s": float(t_3soi_arc),
            "BdotT_km": float(BdotT_arc),
            "BdotR_km": float(BdotR_arc),
            "cov_tr": cov_tr,
            "BdotT_error_km": float(BdotT_arc - 9796.737),
            "BdotR_error_km": float(BdotR_arc - 14970.824),
        }
    )
    print(
        f"Arc Target {d_target:6.1f} Days (Selected {d_sel:7.2f}): "
        f"BdotT={float(BdotT_arc):10.3f} km, "
        f"BdotR={float(BdotR_arc):10.3f} km"
    )

make_bplane_arc_overlay_plot(bplane_records, final_plot_dir)

bplane_summary_rows = []
for rec in bplane_records:
    bplane_summary_rows.append(
        {
            "day_target": rec["day_target"],
            "day_selected": rec["day_selected"],
            "BdotT_km": rec["BdotT_km"],
            "BdotR_km": rec["BdotR_km"],
            "BdotT_error_km": rec["BdotT_error_km"],
            "BdotR_error_km": rec["BdotR_error_km"],
            "sigma_T_km": float(np.sqrt(max(rec["cov_tr"][0, 0], 0.0))),
            "sigma_R_km": float(np.sqrt(max(rec["cov_tr"][1, 1], 0.0))),
        }
    )

pd.DataFrame(bplane_summary_rows).to_csv(final_plot_dir / "bplane_arc_summary.csv", index=False)
print(f"Saved B-Plane Arc Summary: {final_plot_dir / 'bplane_arc_summary.csv'}")
print(f"Saved B-Plane Arc Plot: {final_plot_dir / 'bplane_3sigma_by_arc.png'}")

print(f"Saved Final Test Plots To: {final_plot_dir}")


