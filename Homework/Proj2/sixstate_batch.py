import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.Functions.batch_18_state import batch_estimate_x0
from src.Functions.Ephem import ephem
from src.Functions.jacobians import srp_thirdbody_variational_eq_for6state
from src.Functions.propagation import PropSettings
from src.Functions.srp_dyn_model_6_state import mu_sun_srp_state_deriv_for6state
from src.Functions.stations import Stations
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_state_estimate_3sigma import make_all_state_3sigma_envelope_plot
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.post_processing import run_filter_post_processing


CR_FIXED = 1.38
# Hard-coded run span in days from start of obs file.
# Set to None to run the full arc.
RUN_DAYS = 50

x0_sixstate_batch = np.array(
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
P0_sixstate_batch = np.diag(
    [
        100.0**2,
        100.0**2,
        100.0**2,
        0.1**2,
        0.1**2,
        0.1**2,
    ],
)


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
            radius_earth=radius_earth_km + 0.691750,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "DSS 65",
            lat_deg=40.427222,
            lon_deg=355.749444,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + 0.834539,
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


def limit_measurements_by_run_days(all_meas: list[dict], run_days: float | None):
    """
    Keep measurements from the start of the file through run_days.
    If run_days is None, keep the full arc.
    """
    if len(all_meas) == 0:
        raise ValueError("No measurements found in observation file.")

    t0 = float(all_meas[0]["t"])
    t_last = float(all_meas[-1]["t"])
    if run_days is None:
        return all_meas, t0, t_last

    run_days = float(run_days)
    if run_days <= 0.0:
        raise ValueError("RUN_DAYS must be > 0 or None.")

    t1 = min(t0 + run_days * 86400.0, t_last)
    meas_trim = [m for m in all_meas if float(m["t"]) <= (t1 + 1.0e-12)]
    if len(meas_trim) == 0:
        raise ValueError("Run-days selection produced no measurements.")
    return meas_trim, t0, t1


def build_dyn_and_jac(p_const, sc_const, earth_state_func, sun_state_func):
    dyn = lambda tau, x: mu_sun_srp_state_deriv_for6state(
        t=tau,
        X=x,
        pConst=p_const,
        scConst=sc_const,
        Cr=CR_FIXED,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac(tau, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        r_earth, _ = earth_state_func(tau)
        r_sun, _ = sun_state_func(tau)
        return srp_thirdbody_variational_eq_for6state(
            r_sc=x[0:3],
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

    return dyn, jac


def batch_info_to_filter_like_out(info: dict, R_km: np.ndarray) -> dict:
    xhat = np.asarray(info["state_hist"], dtype=float)
    P = np.asarray(info["P_hist"], dtype=float)
    two_sigma = 2.0 * np.sqrt(np.maximum(np.diagonal(P, axis1=1, axis2=2), 0.0))

    prefit = np.asarray(info.get("prefit_residuals", np.full((xhat.shape[0], 2), np.nan)), dtype=float)
    postfit_nl = np.asarray(info.get("postfit_residuals", np.full((xhat.shape[0], 2), np.nan)), dtype=float)

    postfit_lin = None
    if "postfit_resids_linear_hist" in info and len(info["postfit_resids_linear_hist"]) > 0:
        postfit_lin = np.asarray(info["postfit_resids_linear_hist"][-1], dtype=float)
    if postfit_lin is None:
        postfit_lin = postfit_nl.copy()

    return {
        "t_meas": np.asarray(info["t_meas"], dtype=float),
        "station_meas": list(info["station_meas"]),
        "xhat_meas": xhat,
        "Xhat_meas": xhat,
        "P_meas": P,
        "two_sigma_meas": two_sigma,
        "prefit_resids_final": prefit,
        "postfit_resids_linear_final": postfit_lin,
        "postfit_resids_meas": postfit_nl,
        "state_error_meas": None,
        "R": np.asarray(R_km, dtype=float),
        "rms_final": None,
        "rms_by_iter": None,
    }


def to_plotting_result_6state_from_filter_km(out: dict, *, R_km: np.ndarray):
    xhat = np.asarray(out["xhat_meas"], dtype=float)
    P = np.asarray(out["P_meas"], dtype=float)

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
        "prefit_resids_final": np.asarray(out["prefit_resids_final"], dtype=float) * 1000.0,
        "postfit_resids_linear_final": np.asarray(out["postfit_resids_linear_final"], dtype=float) * 1000.0,
        "postfit_resids_meas": np.asarray(out["postfit_resids_meas"], dtype=float) * 1000.0,
        "rms_final": out.get("rms_final", {}),
        "rms_by_iter": out.get("rms_by_iter", None),
        "R": np.asarray(R_km, dtype=float) * (1000.0**2),
    }
    return run_filter_post_processing(out=d)


def make_standard_plot_set(out: dict, outdir: Path, *, R_km: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    result = to_plotting_result_6state_from_filter_km(out, R_km=R_km)
    make_postfit_residuals_linear_plot(result, outdir)
    make_trace_cov_pos_vel_plot(result, outdir, length_unit="m")
    make_all_state_3sigma_envelope_plot(out, outdir)


def print_postfit_3sigma_coverage(out: dict, R_km: np.ndarray, *, label: str):
    postfit = out.get("postfit_resids_linear_final", None)
    if postfit is None:
        postfit = out.get("postfit_resids_meas", None)
    if postfit is None:
        print(f"{label}: no postfit residuals available for 3-sigma coverage.")
        return

    pf = np.asarray(postfit, dtype=float)
    if pf.ndim != 2 or pf.shape[1] < 2:
        print(f"{label}: postfit residual shape invalid for 3-sigma coverage: {pf.shape}")
        return

    R = np.asarray(R_km, dtype=float)
    sig_rho = float(np.sqrt(max(R[0, 0], 0.0)))
    sig_rhod = float(np.sqrt(max(R[1, 1], 0.0)))
    b_rho = 3.0 * sig_rho
    b_rhod = 3.0 * sig_rhod

    valid_rho = np.isfinite(pf[:, 0])
    valid_rhod = np.isfinite(pf[:, 1])
    in_rho = valid_rho & (np.abs(pf[:, 0]) <= b_rho)
    in_rhod = valid_rhod & (np.abs(pf[:, 1]) <= b_rhod)

    n_rho = int(np.sum(valid_rho))
    n_rhod = int(np.sum(valid_rhod))
    pct_rho = (100.0 * float(np.sum(in_rho)) / n_rho) if n_rho > 0 else np.nan
    pct_rhod = (100.0 * float(np.sum(in_rhod)) / n_rhod) if n_rhod > 0 else np.nan

    print(f"\n{label} 3-sigma postfit coverage:")
    print(f"  Range       : {pct_rho:6.2f}% ({int(np.sum(in_rho))}/{n_rho}) within +/-3sigma")
    print(f"  Range-Rate  : {pct_rhod:6.2f}% ({int(np.sum(in_rhod))}/{n_rhod}) within +/-3sigma")


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
    np.savez_compressed(
        save_path,
        t_meas=np.asarray(out["t_meas"], dtype=float),
        station_meas=np.asarray(out["station_meas"], dtype=str),
        xhat_meas=np.asarray(out["xhat_meas"], dtype=float),
        P_meas=np.asarray(out["P_meas"], dtype=float),
        prefit_resids_final=np.asarray(out.get("prefit_resids_final", np.empty((0, 2))), dtype=float),
        postfit_resids_linear_final=np.asarray(
            out.get("postfit_resids_linear_final", np.empty((0, 2))),
            dtype=float,
        ),
    )


def save_final_summary_json(out: dict, save_path: Path, history_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x_fwd = np.asarray(out["xhat_meas"][-1], dtype=float).reshape(-1)
    P_fwd = np.asarray(out["P_meas"][-1], dtype=float)
    t_fwd = float(np.asarray(out["t_meas"], dtype=float)[-1])

    payload = {
        "history_source": str(history_path.resolve()),
        "units": {
            "state": "[km, km, km, km/s, km/s, km/s]",
            "covariance": "km-based full-state covariance; position variances in km^2, velocity variances in (km/s)^2, and cross terms in consistent mixed units",
            "time": "seconds since epoch",
            "fixed_Cr": "unitless",
        },
        "sixstate_batch": {
            "fixed_Cr": float(CR_FIXED),
            "final_state_batch_forward": x_fwd.tolist(),
            "final_cov_batch_forward": P_fwd.tolist(),
            "final_cov_batch_forward_full_km": P_fwd.tolist(),
            "final_time_s_batch_forward": t_fwd,
        },
        # Compatibility block for part3 scripts that read maneuver_iekf keys.
        "maneuver_iekf": {
            "final_state_iekf_forward": x_fwd.tolist(),
            "final_cov_iekf_forward": P_fwd.tolist(),
            "final_cov_iekf_forward_full_km": P_fwd.tolist(),
            "final_time_s_iekf_forward": t_fwd,
            "final_state_iekf_smoothed": None,
            "final_cov_iekf_smoothed": None,
            "final_cov_iekf_smoothed_full_km": None,
            "final_time_s_iekf_smoothed": None,
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run full-arc six-state iterated batch with fixed Cr.")
    parser.add_argument("--max-iter", type=int, default=10, help="Max batch iterations.")
    parser.add_argument("--tol", type=float, default=1.0e-8, help="Batch convergence tolerance.")
    args = parser.parse_args()

    obs_path = Path(__file__).resolve().parent / "Given_data" / "Project2b_Obs.txt"
    all_meas_full = load_project2_obs(obs_path)
    all_meas, t0_run, t1_run = limit_measurements_by_run_days(all_meas_full, RUN_DAYS)
    stations = build_stations()

    p_const, sc_const, earth_state_func, sun_state_func = build_problem_constants()
    dyn_fun, dyn_jac = build_dyn_and_jac(p_const, sc_const, earth_state_func, sun_state_func)

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_batch = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    x0_hat, P0_hat, info = batch_estimate_x0(
        all_meas=all_meas,
        stations=stations,
        x0_bar=x0_sixstate_batch,
        P0=P0_sixstate_batch,
        R=R_batch,
        dyn_fun=dyn_fun,
        dyn_jac=dyn_jac,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        prop_settings=PropSettings(rtol=1.0e-10, atol=1.0e-10, method="RK45"),
    )
    out = batch_info_to_filter_like_out(info, R_km=R_batch)

    plot_root = Path(__file__).resolve().parent / "Plots" / "Sixstate Batch"
    chunk_root = plot_root / "Chunks"
    total_root = plot_root / "Total Run"
    chunk_root.mkdir(parents=True, exist_ok=True)
    total_root.mkdir(parents=True, exist_ok=True)

    history_path = plot_root / "sixstate_batch_history.npz"
    save_history(out, history_path)
    final_json_path = plot_root / "final_states_sixstate_batch.json"
    save_final_summary_json(out, final_json_path, history_path)

    make_standard_plot_set(out, total_root, R_km=R_batch)
    print_postfit_3sigma_coverage(out, R_batch, label="Forward Sixstate Batch")

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
            make_standard_plot_set(out_chunk, chunk_dir, R_km=R_batch)
            print(
                f"Saved Chunk {chunk_i:02d} Plots: Days {day_start:.3f} To {day_end:.3f} "
                f"({len(out_chunk['t_meas'])} measurements)"
            )
        day_start = day_end
        chunk_i += 1

    xf = np.asarray(out["xhat_meas"][-1], dtype=float).reshape(-1)
    print("\nFinal State Estimate (Sixstate Batch):")
    print(f"  Position [km]    : [{xf[0]:.6f}, {xf[1]:.6f}, {xf[2]:.6f}]")
    print(f"  Velocity [km/s]  : [{xf[3]:.9f}, {xf[4]:.9f}, {xf[5]:.9f}]")
    print(f"  Fixed Cr [-]     : {CR_FIXED:.9f}")
    print(f"  Batch Iterations : {info.get('num_iters', 'n/a')}")
    if RUN_DAYS is None:
        print("  Run Span         : Full data arc")
    else:
        print(
            f"  Run Span         : First {float(RUN_DAYS):.3f} days "
            f"(actual {t0_run/86400.0:.3f} -> {t1_run/86400.0:.3f} days)"
        )
    print(f"\nSaved Sixstate Batch Plots To: {plot_root}")
    print(f"Saved Sixstate Batch Final-State JSON To: {final_json_path}")
    print(f"Saved Sixstate Batch Time-Step History To: {history_path}")

    postfit_range = out["postfit_resids_meas"][:, 0]
    t_arr = out["t_meas"]
    st_arr = out["station_meas"]
    worst_idx = np.nanargmax(np.abs(postfit_range))
    print("Worst range residual:")
    print(f"  Index   : {worst_idx}")
    print(f"  Time    : {t_arr[worst_idx]:.1f} s = day {t_arr[worst_idx] / 86400.0:.4f}")
    print(f"  Station : {st_arr[worst_idx]}")
    print(f"  Residual: {postfit_range[worst_idx] * 1000.0:.2f} m")

    # Keep these in scope for quick debugging if needed.
    _ = (x0_hat, P0_hat)


if __name__ == "__main__":
    main()
