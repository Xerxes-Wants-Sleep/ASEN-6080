import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.Functions.iekf_maneuver_check import IEKF2
from src.Functions.propagation import PropSettings
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model_6_state import mu_sun_srp_state_deriv_for6state
from src.Functions.jacobians import srp_thirdbody_variational_eq_for6state
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.plot_state_estimate_3sigma import make_all_state_3sigma_envelope_plot
from src.helpers.plotting.plot_state_errors_18 import make_state_errors_18_plot
from src.helpers.plotting.common import as_hours, savefig


CR_FIXED = 1.38
W_EARTH_RAD_PER_S = 7.29211585275553e-5
MANEUVER_DAY_EST = 217

# Station-estimation controls
FIXED_DSS_NAMES = ("DSS 34",)
# Backward-compatible alias used in some metadata fields.
FIXED_DSS_NAME = FIXED_DSS_NAMES[0]
FIXED_DSS_VAR = 1.0e-12
FREE_DSS_VAR = 1000000

ENABLE_PLOT_OUTLIER_MASK = True
PLOT_OUTLIER_ABS_THRESHOLD_M = 700.0

x0_rv = np.array(
    [
        -274096770.76544,
        -92859266.4499061,
        -40199493.6677441,
        32.6704564599943,
        -8.93838913761049,
        -3.87881914050316,
    ],
    dtype=float,
)


def build_stations():
    theta0_deg = 0
    radius_earth_km = 6378.1363
    return [
        Stations(
            "DSS 34",
            lat_deg=-35.398333,
            lon_deg=148.981944,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + 0.691750,
            w_earth_rad_per_s=W_EARTH_RAD_PER_S,
        ),
        Stations(
            "DSS 65",
            lat_deg=40.427222,
            lon_deg=355.749444,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + 0.834539,
            w_earth_rad_per_s=W_EARTH_RAD_PER_S,
        ),
        Stations(
            "DSS 13",
            lat_deg=35.247164,
            lon_deg=243.205000,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + 1.07114904,
            w_earth_rad_per_s=W_EARTH_RAD_PER_S,
        ),
    ]


def build_problem_constants():
    jd0 = 2456296.25
    mu_sun = 132712440017.987
    au_km = 149597870.7
    solar_flux_w_m2 = 1357.0
    srp_area_mass_ratio = 0.01

    p_const = type("pConst", (), {})()
    p_const.mu_earth = 3.98600432896939e5
    p_const.mu_sun = mu_sun

    sc_const = type("scConst", (), {})()
    sc_const.area = srp_area_mass_ratio
    sc_const.mass = 1.0
    sc_const.solar_flux_1au = solar_flux_w_m2
    sc_const.c = 299792458.0
    sc_const.AU_m = au_km * 1000.0
    sc_const.Cr_fixed = CR_FIXED

    def earth_state_func(tau):
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    def sun_state_func(tau):
        jd = jd0 + float(tau) / 86400.0
        r_e_km, v_e_km_s, _ = ephem(jd, 3, frame="EME2000")
        r_e_km = np.asarray(r_e_km, dtype=float).reshape(3,)
        v_e_km_s = np.asarray(v_e_km_s, dtype=float).reshape(3,)
        return -r_e_km, -v_e_km_s

    return p_const, sc_const, earth_state_func, sun_state_func


def load_project2_obs(obs_path: Path) -> list[dict]:
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


def build_station_augmented_initials(stations: list[Stations]):
    station_state_map = {st.name: i for i, st in enumerate(stations)}
    x_station_blocks = []
    p_station_diag = []
    station_nominal = {}
    station_geo_meta = {}

    for st in stations:
        r_eci_km, _ = st.station_eci(0.0)
        r_eci_km = np.asarray(r_eci_km, dtype=float).reshape(3,)
        station_nominal[st.name] = r_eci_km.copy()
        station_geo_meta[st.name] = {
            "lat_deg": float(st.lat_deg),
            "lon_deg": float(st.lon_deg),
            "radius_km": float(st.radius_earth),
            "theta0_deg": float(st.theta0_deg),
        }
        x_station_blocks.append(r_eci_km)

        var_i = FIXED_DSS_VAR if st.name in FIXED_DSS_NAMES else FREE_DSS_VAR
        p_station_diag.extend([float(var_i)] * 3)

    # 18-state layout: [r(3), v(3), filler(3), stations(9)]
    x0 = np.hstack((x0_rv, np.zeros(3, dtype=float), np.hstack(x_station_blocks)))
    p0 = np.zeros((18, 18), dtype=float)
    p0[:6, :6] = np.diag([100.0**2, 100.0**2, 100.0**2, 0.1**2, 0.1**2, 0.1**2])
    p0[6:9, 6:9] = np.diag([1.0e-4**2, 1.0e-4**2, 1.0e-4**2])
    p0[9:, 9:] = np.diag(np.asarray(p_station_diag, dtype=float))
    return x0, p0, station_state_map, station_nominal, station_geo_meta


def _wrap_lon_deg_0_360(lon_deg: float) -> float:
    return float(np.mod(float(lon_deg), 360.0))


def _wrap_lon_delta_deg_pm180(delta_lon_deg: float) -> float:
    return float((float(delta_lon_deg) + 180.0) % 360.0 - 180.0)


def eci_station_to_lat_lon_radius_deg(r_eci_km: np.ndarray, *, theta0_deg: float) -> tuple[float, float, float]:
    r_eci = np.asarray(r_eci_km, dtype=float).reshape(3,)
    th = np.deg2rad(float(theta0_deg))
    c = float(np.cos(th))
    s = float(np.sin(th))

    # station_eci = R3(theta0) @ r_ecef at t=0, so invert with R3(-theta0)
    r_ecef = np.array(
        [
            c * r_eci[0] + s * r_eci[1],
            -s * r_eci[0] + c * r_eci[1],
            r_eci[2],
        ],
        dtype=float,
    )
    x, y, z = [float(v) for v in r_ecef]
    rho_xy = float(np.hypot(x, y))
    radius_km = float(np.linalg.norm(r_ecef))
    lat_deg = float(np.rad2deg(np.arctan2(z, rho_xy)))
    lon_deg = _wrap_lon_deg_0_360(float(np.rad2deg(np.arctan2(y, x))))
    return lat_deg, lon_deg, radius_km


def _skew(omega_vec: np.ndarray) -> np.ndarray:
    wx, wy, wz = [float(v) for v in np.asarray(omega_vec, dtype=float).reshape(3)]
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=float,
    )


def state_deriv_18state_with_stations(
    t: float,
    X: np.ndarray,
    p_const,
    sc_const,
    earth_state_func,
    sun_state_func,
):
    X = np.asarray(X, dtype=float).reshape(18,)
    dX = np.zeros(18, dtype=float)

    dX[:6] = mu_sun_srp_state_deriv_for6state(
        t=t,
        X=X[:6],
        pConst=p_const,
        scConst=sc_const,
        Cr=CR_FIXED,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )
    dX[6:9] = 0.0

    omega = np.array([0.0, 0.0, W_EARTH_RAD_PER_S], dtype=float)
    for i in range(3):
        base = 9 + 3 * i
        dX[base:base + 3] = np.cross(omega, X[base:base + 3])
    return dX


def jacobian_18state_with_stations(
    t: float,
    X: np.ndarray,
    p_const,
    sc_const,
    earth_state_func,
    sun_state_func,
):
    X = np.asarray(X, dtype=float).reshape(18,)
    r_earth, _ = earth_state_func(t)
    r_sun, _ = sun_state_func(t)

    A6 = srp_thirdbody_variational_eq_for6state(
        r_sc=X[0:3],
        r_earth=r_earth,
        r_sun=r_sun,
        Cr=CR_FIXED,
        area=sc_const.area,
        mass=sc_const.mass,
        mu_earth=p_const.mu_earth,
        mu_i=p_const.mu_sun,
        solar_flux_1au=sc_const.solar_flux_1au,
        c=sc_const.c,
        AU_m=sc_const.AU_m,
    )

    A = np.zeros((18, 18), dtype=float)
    A[:6, :6] = A6

    A_gs = _skew(np.array([0.0, 0.0, W_EARTH_RAD_PER_S], dtype=float))
    for i in range(3):
        base = 9 + 3 * i
        A[base:base + 3, base:base + 3] = A_gs
    return A


def to_plotting_result_6state_from_filter_km(out: dict, *, R_km: np.ndarray):
    xhat_key = "xhat_meas" if "xhat_meas" in out else "Xhat_meas"
    phat_key = "P_meas" if "P_meas" in out else "Phat_meas"

    xhat = np.asarray(out[xhat_key], dtype=float)
    P = np.asarray(out[phat_key], dtype=float)

    xhat6_m = xhat[:, 0:6] * 1000.0
    P6_m = P[:, 0:6, 0:6] * (1000.0**2)
    two_sigma6_m = 2.0 * np.sqrt(np.maximum(np.diagonal(P6_m, axis1=1, axis2=2), 0.0))

    d = {
        "t_meas": np.asarray(out["t_meas"], dtype=float),
        "station_meas": list(out["station_meas"]),
        "xhat_meas": xhat6_m,
        "Xhat_meas": xhat6_m,
        "P_meas": P6_m,
        "two_sigma_meas": two_sigma6_m,
        "state_error_meas": None,
        "prefit_resids_final": None
        if out.get("prefit_resids_final", None) is None
        else np.asarray(out["prefit_resids_final"], dtype=float) * 1000.0,
        "postfit_resids_linear_final": None
        if out.get("postfit_resids_linear_final", None) is None
        else np.asarray(out["postfit_resids_linear_final"], dtype=float) * 1000.0,
        "postfit_resids_meas": None
        if out.get("postfit_resids_meas", None) is None
        else np.asarray(out["postfit_resids_meas"], dtype=float) * 1000.0,
        "rms_final": out.get("rms_final", {}),
        "rms_by_iter": out.get("rms_by_iter", None),
        "R": np.asarray(R_km, dtype=float) * (1000.0**2),
    }
    return run_filter_post_processing(out=d)


def drop_single_range_outlier_for_plot(out: dict, *, abs_threshold_m: float = 700.0) -> dict:
    out_plot = dict(out)
    postfit_lin = out.get("postfit_resids_linear_final", None)
    if postfit_lin is None:
        return out_plot

    pf_lin = np.asarray(postfit_lin, dtype=float).copy()
    if pf_lin.ndim != 2 or pf_lin.shape[1] < 1:
        return out_plot

    range_res_m = pf_lin[:, 0] * 1000.0
    valid = np.isfinite(range_res_m)
    if not np.any(valid):
        return out_plot

    valid_idx = np.where(valid)[0]
    idx = int(valid_idx[np.argmax(np.abs(range_res_m[valid]))])
    peak_m = float(range_res_m[idx])
    if np.abs(peak_m) < float(abs_threshold_m):
        return out_plot

    pf_lin[idx, :] = np.nan
    out_plot["postfit_resids_linear_final"] = pf_lin

    postfit_nl = out.get("postfit_resids_meas", None)
    if postfit_nl is not None:
        pf_nl = np.asarray(postfit_nl, dtype=float).copy()
        if pf_nl.shape == pf_lin.shape:
            pf_nl[idx, :] = np.nan
            out_plot["postfit_resids_meas"] = pf_nl

    return out_plot


def make_standard_plot_set(out: dict, outdir: Path, *, R_km: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    out_plot = (
        drop_single_range_outlier_for_plot(out, abs_threshold_m=PLOT_OUTLIER_ABS_THRESHOLD_M)
        if ENABLE_PLOT_OUTLIER_MASK
        else out
    )
    # Apply the outlier mask specifically to residual plotting.
    result_postfit = to_plotting_result_6state_from_filter_km(out_plot, R_km=R_km)
    make_postfit_residuals_linear_plot(result_postfit, outdir)

    # Keep covariance/state-envelope plots based on the unmasked run output.
    result_nominal = to_plotting_result_6state_from_filter_km(out, R_km=R_km)
    make_trace_cov_pos_vel_plot(result_nominal, outdir, length_unit="m")
    make_all_state_3sigma_envelope_plot(out, outdir)


def make_total_state_error_plot(
    out: dict,
    outdir: Path,
    *,
    P0: np.ndarray,
    filename: str = "state_error_total_normalized.png",
):
    state_err = out.get("state_error_meas", None)
    if state_err is None:
        return False

    e = np.asarray(state_err, dtype=float)
    if e.ndim != 2 or e.shape[1] < 18:
        return False

    make_state_errors_18_plot(
        t_meas=np.asarray(out["t_meas"], dtype=float),
        state_err=e[:, :18] * 1000.0,
        outdir=outdir,
        title="State Errors (18-state, Forward)",
        filename="state_errors_18_forward.png",
    )

    t_hr = as_hours(np.asarray(out["t_meas"], dtype=float))
    sig0 = np.sqrt(np.maximum(np.diag(np.asarray(P0, dtype=float)), 1.0e-30)).reshape(1, -1)
    e_norm = e[:, :18] / sig0[:, :18]
    total_norm = np.linalg.norm(e_norm, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t_hr, total_norm, ".", markersize=2)
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("||state error|| (normalized by sqrt(diag(P0)))")
    ax.set_title("Total Normalized State Error")
    savefig(fig, outdir / filename, show=False)
    return True


def print_station_estimates(
    X_final,
    P_final,
    station_state_map,
    station_nominal,
    station_geo_meta,
    *,
    label: str,
    t_final_s: float = 0.0,
):
    print(f"\n{label} station estimates:")
    for name, idx in station_state_map.items():
        base = 9 + 3 * int(idx)
        r_est = np.asarray(X_final[base:base + 3], dtype=float)
        cov = np.asarray(P_final[base:base + 3, base:base + 3], dtype=float)
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
        r_nom = np.asarray(station_nominal[name], dtype=float)
        geo_nom = station_geo_meta[name]
        theta_final_deg = float(geo_nom["theta0_deg"]) + float(np.rad2deg(W_EARTH_RAD_PER_S * t_final_s))
        lat_est, lon_est, rad_est = eci_station_to_lat_lon_radius_deg(
            r_est,
            theta0_deg=theta_final_deg,
        )
        dlat_deg = float(lat_est - float(geo_nom["lat_deg"]))
        dlon_deg = _wrap_lon_delta_deg_pm180(lon_est - float(geo_nom["lon_deg"]))
        dr_rad_m = float((rad_est - float(geo_nom["radius_km"])) * 1000.0)
        dr_m = (r_est - r_nom) * 1000.0
        print(
            f"  {name}: r_eci_km=[{r_est[0]:.6f}, {r_est[1]:.6f}, {r_est[2]:.6f}], "
            f"sigma_km=[{sig[0]:.3e}, {sig[1]:.3e}, {sig[2]:.3e}], "
            f"delta_m=[{dr_m[0]:.3f}, {dr_m[1]:.3f}, {dr_m[2]:.3f}], "
            f"lat_lon_radius=[{lat_est:.6f} deg, {lon_est:.6f} deg, {rad_est:.6f} km], "
            f"delta_geo=[{dlat_deg:+.6e} deg, {dlon_deg:+.6e} deg, {dr_rad_m:+.3f} m]"
        )


def slice_out_by_day_window(out: dict, day_start: float, day_end: float, *, include_start: bool) -> dict | None:
    t_meas = np.asarray(out["t_meas"], dtype=float)
    t_days = t_meas / 86400.0
    if include_start:
        mask = (t_days >= float(day_start)) & (t_days <= float(day_end) + 1.0e-12)
    else:
        mask = (t_days > float(day_start)) & (t_days <= float(day_end) + 1.0e-12)

    idx = np.where(mask)[0]
    if idx.size == 0:
        return None

    n_total = len(t_meas)

    def _slice_value(v):
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            if v.ndim >= 1 and v.shape[0] == n_total:
                return v[idx]
            return v
        if isinstance(v, list):
            if len(v) == n_total:
                return [v[i] for i in idx]
            return v
        return v

    out_slice = {}
    for k, v in out.items():
        out_slice[k] = _slice_value(v)
    return out_slice


def save_history(out: dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "t_meas": np.asarray(out["t_meas"], dtype=float),
        "station_meas": np.asarray(out["station_meas"], dtype=str),
        "xhat_meas": np.asarray(out["xhat_meas"], dtype=float),
        "P_meas": np.asarray(out["P_meas"], dtype=float),
        "prefit_resids_final": np.asarray(out.get("prefit_resids_final", np.empty((0, 2))), dtype=float),
        "postfit_resids_linear_final": np.asarray(
            out.get("postfit_resids_linear_final", np.empty((0, 2))),
            dtype=float,
        ),
    }
    np.savez_compressed(save_path, **payload)


def save_final_summary_json(
    out: dict,
    save_path: Path,
    *,
    station_state_map: dict,
    station_nominal: dict,
    station_geo_meta: dict,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x_fwd = np.asarray(out["xhat_meas"][-1], dtype=float).reshape(-1)
    P_fwd = np.asarray(out["P_meas"][-1], dtype=float)
    t_fwd = float(np.asarray(out["t_meas"], dtype=float)[-1])

    def _pack_station_block(x_vec: np.ndarray, P_mat: np.ndarray, t_s: float):
        block = {}
        for name, idx in station_state_map.items():
            base = 9 + 3 * int(idx)
            r_est = np.asarray(x_vec[base:base + 3], dtype=float)
            cov = np.asarray(P_mat[base:base + 3, base:base + 3], dtype=float)
            sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
            r_nom = np.asarray(station_nominal[name], dtype=float)
            geo_nom = station_geo_meta[name]
            theta_final_deg = float(geo_nom["theta0_deg"]) + float(np.rad2deg(W_EARTH_RAD_PER_S * t_s))
            lat_est, lon_est, rad_est = eci_station_to_lat_lon_radius_deg(
                r_est,
                theta0_deg=theta_final_deg,
            )
            dlat_deg = float(lat_est - float(geo_nom["lat_deg"]))
            dlon_deg = _wrap_lon_delta_deg_pm180(lon_est - float(geo_nom["lon_deg"]))
            dr_rad_m = float((rad_est - float(geo_nom["radius_km"])) * 1000.0)
            dr_m = (r_est - r_nom) * 1000.0
            block[name] = {
                "r_eci_km": r_est.tolist(),
                "sigma_xyz_km": sig.tolist(),
                "delta_from_nominal_m": dr_m.tolist(),
                "input_station_location": {
                    "lat_deg": float(geo_nom["lat_deg"]),
                    "lon_deg": float(geo_nom["lon_deg"]),
                    "radius_km": float(geo_nom["radius_km"]),
                },
                "estimated_station_location": {
                    "lat_deg": float(lat_est),
                    "lon_deg": float(lon_est),
                    "radius_km": float(rad_est),
                },
                "delta_from_input_location": {
                    "dlat_deg": float(dlat_deg),
                    "dlon_deg_wrapped_pm180": float(dlon_deg),
                    "dradius_m": float(dr_rad_m),
                },
            }
        return block

    payload = {
        "units": {
            "state": "[r,v,filler,stations] in km and km/s",
            "covariance": "km-based full-state covariance with mixed-unit cross terms",
            "time": "seconds since epoch",
        },
        "station_iekf": {
            "fixed_Cr": float(CR_FIXED),
            "assumed_maneuver_day": float(MANEUVER_DAY_EST),
            "fixed_dss_name": FIXED_DSS_NAME,
            "fixed_dss_names": list(FIXED_DSS_NAMES),
            "final_state_forward": x_fwd.tolist(),
            "final_cov_forward": P_fwd.tolist(),
            "final_time_s_forward": t_fwd,
        },
        # Compatibility block for existing Part 3 readers.
        "maneuver_iekf": {
            "final_state_iekf_forward": x_fwd.tolist(),
            "final_cov_iekf_forward": P_fwd.tolist(),
            "final_cov_iekf_forward_full_km": P_fwd.tolist(),
            "final_time_s_iekf_forward": t_fwd,
        },
        "estimated_stations": {
            "fixed_dss_name": FIXED_DSS_NAME,
            "fixed_dss_names": list(FIXED_DSS_NAMES),
            "forward": _pack_station_block(x_fwd, P_fwd, t_fwd),
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    obs_path = Path(__file__).resolve().parent / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs(obs_path)
    stations = build_stations()
    x0, P0, station_state_map, station_nominal, station_geo_meta = build_station_augmented_initials(stations)

    p_const, sc_const, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    sigma_acc_km_s2 = 1.0e-9
    Q = np.diag([sigma_acc_km_s2**2, sigma_acc_km_s2**2, sigma_acc_km_s2**2])

    dyn = lambda tau, x: state_deriv_18state_with_stations(
        t=tau,
        X=x,
        p_const=p_const,
        sc_const=sc_const,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    jac = lambda tau, x: jacobian_18state_with_stations(
        t=tau,
        X=x,
        p_const=p_const,
        sc_const=sc_const,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    iekf = IEKF2(
        x0=x0,
        P0=P0,
        R=R,
        Q=Q,
        dyn_fun=dyn,
        dyn_jac=jac,
        station_state_map=station_state_map,
        station_start_index=9,
        num_stations=3,
        omega_vec=np.array([0.0, 0.0, W_EARTH_RAD_PER_S], dtype=float),
        prop_settings=PropSettings(rtol=1.0e-10, atol=1.0e-10, method="RK45"),
        first_pass_gap_s=6 * 3600.0,
    )

    out = iekf.run(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=None,
        iterated=True,
        max_iter=10,
        iter_tol=1.0e-10,
        bound_level=5.0,
        no_snc_on_first_after_gap=True,
        gap_threshold_s=6 * 3600.0,
        show_progress=True,
        progress_every=50,
        t_man=MANEUVER_DAY_EST * 86400.0,
    )
    print(
        f"Station IEKF Run Complete. Number Of Updates: {len(out['t_meas'])} "
        f"(Fixed DSS={', '.join(FIXED_DSS_NAMES)}, Assumed Maneuver Day={MANEUVER_DAY_EST:.3f})"
    )

    plot_root = Path(__file__).resolve().parent / "Plots" / "Station IEKF"
    chunk_root = plot_root / "Chunks"
    total_root = plot_root / "Total Run"
    chunk_root.mkdir(parents=True, exist_ok=True)
    total_root.mkdir(parents=True, exist_ok=True)

    history_path = plot_root / "station_iekf_history.npz"
    save_history(out, history_path)
    final_json_path = plot_root / "final_states_station_iekf.json"
    save_final_summary_json(
        out,
        final_json_path,
        station_state_map=station_state_map,
        station_nominal=station_nominal,
        station_geo_meta=station_geo_meta,
    )

    make_standard_plot_set(out, total_root, R_km=R)
    made_total_state_error = make_total_state_error_plot(out, total_root, P0=P0)

    t_days = np.asarray(out["t_meas"], dtype=float) / 86400.0
    max_day = float(np.max(t_days))

    day_start = 0.0
    chunk_i = 1
    while day_start < (max_day - 1.0e-12):
        day_end = min(day_start + 50.0, max_day)
        out_chunk = slice_out_by_day_window(
            out,
            day_start,
            day_end,
            include_start=(chunk_i == 1),
        )
        if out_chunk is not None:
            d0 = int(np.floor(day_start + 1.0e-9))
            d1 = int(np.ceil(day_end - 1.0e-9))
            chunk_dir = chunk_root / f"Days_{d0:03d}_{d1:03d}"
            make_standard_plot_set(out_chunk, chunk_dir, R_km=R)
            print(
                f"Saved Chunk {chunk_i:02d} Plots: Days {day_start:.3f} To {day_end:.3f} "
                f"({len(out_chunk['t_meas'])} measurements)"
            )
        day_start = day_end
        chunk_i += 1

    xf = np.asarray(out["xhat_meas"][-1], dtype=float).reshape(-1)
    Pf = np.asarray(out["P_meas"][-1], dtype=float)
    print("\nFinal State Estimate (Forward Station IEKF):")
    print(f"  Position [km]    : [{xf[0]:.6f}, {xf[1]:.6f}, {xf[2]:.6f}]")
    print(f"  Velocity [km/s]  : [{xf[3]:.9f}, {xf[4]:.9f}, {xf[5]:.9f}]")
    print(f"  Fixed Cr [-]     : {CR_FIXED:.9f}")
    print(f"  Assumed Maneuver Day: {MANEUVER_DAY_EST:.6f}")
    print(f"  Sigma Free: {FREE_DSS_VAR:.6f}")
    print_station_estimates(
        xf,
        Pf,
        station_state_map,
        station_nominal,
        station_geo_meta,
        label="Forward IEKF",
        t_final_s=float(out["t_meas"][-1]),
    )

    print(f"\nSaved Station IEKF Plots To: {plot_root}")
    if not made_total_state_error:
        print("State-error total plot not generated (no truth-based state_error_meas provided).")
    print(f"Saved Station IEKF Final-State JSON To: {final_json_path}")
    print(f"Saved Station IEKF Time-Step History To: {history_path}")


if __name__ == "__main__":
    main()
