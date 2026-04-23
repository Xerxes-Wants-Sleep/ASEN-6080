import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.Functions.itekf_2 import IEKF2
from src.Functions.propagation import PropSettings
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model_6_state import mu_sun_srp_state_deriv_for6state
from src.Functions.jacobians import srp_thirdbody_variational_eq_for6state
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.plot_state_estimate_3sigma import make_all_state_3sigma_envelope_plot


CR_FIXED = 1.38

# Stop-start configuration requested by user.
STAGE1_START_DAY = 1.0
BREAK_INTERVAL_DAYS = 50.0
COVARIANCE_INFLATION_FACTOR = 25
# Extra explicit break(s) in days, in addition to periodic breaks.
ADDITIONAL_BREAK_DAYS = [217.0]

x0_iekf = np.array(
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
P0_iekf = np.diag(
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


def split_measurements_periodic(
    all_meas: list[dict],
    *,
    stage1_start_day: float,
    break_interval_days: float,
    additional_break_days: list[float] | None = None,
) -> tuple[list[list[dict]], list[float], float]:
    all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
    if len(all_meas) == 0:
        raise ValueError("No measurements found.")
    if break_interval_days <= 0.0:
        raise ValueError(f"break_interval_days must be > 0; got {break_interval_days}")

    t_epoch = float(all_meas[0]["t"])
    t0 = float(stage1_start_day) * 86400.0
    meas_from_start = [m for m in all_meas if float(m["t"]) >= (t0 - 1.0e-12)]
    if len(meas_from_start) == 0:
        raise ValueError(
            f"No measurements found at/after start day {stage1_start_day}."
        )

    t_end_day = max(float(m["t"]) for m in meas_from_start) / 86400.0
    # Breaks align on global 50-day boundaries: 50, 100, 150, ...
    first_break_day = float(break_interval_days) * float(np.ceil(stage1_start_day / break_interval_days))
    break_days: list[float] = []
    d = first_break_day
    while d < (t_end_day - 1.0e-12):
        break_days.append(float(d))
        d += float(break_interval_days)

    if additional_break_days is not None:
        for bd in additional_break_days:
            bd = float(bd)
            if (bd > stage1_start_day + 1.0e-12) and (bd < t_end_day - 1.0e-12):
                break_days.append(bd)

    # Deduplicate and sort.
    break_days = sorted(set(break_days))

    stages: list[list[dict]] = []
    prev_day = float(stage1_start_day)
    for i, bday in enumerate(break_days):
        if i == 0:
            stage = [
                m
                for m in meas_from_start
                if (float(m["t"]) / 86400.0 >= prev_day - 1.0e-12)
                and (float(m["t"]) / 86400.0 <= bday + 1.0e-12)
            ]
        else:
            stage = [
                m
                for m in meas_from_start
                if (float(m["t"]) / 86400.0 > prev_day + 1.0e-12)
                and (float(m["t"]) / 86400.0 <= bday + 1.0e-12)
            ]
        if len(stage) == 0:
            raise ValueError(f"Stage window ({prev_day:.3f}, {bday:.3f}] has no measurements.")
        stages.append(stage)
        prev_day = float(bday)

    final_stage = [
        m
        for m in meas_from_start
        if (float(m["t"]) / 86400.0 > prev_day + 1.0e-12)
    ]
    if len(final_stage) > 0:
        stages.append(final_stage)

    if len(stages) == 0:
        raise ValueError("Periodic stop-start stage split produced no stages.")

    return stages, break_days, t_epoch


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
        "state_error_meas": None
        if out.get("state_error_meas", None) is None
        else np.asarray(out["state_error_meas"], dtype=float)[:, 0:6] * 1000.0,
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


def make_standard_plot_set(out: dict, outdir: Path, *, R_km: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    result = to_plotting_result_6state_from_filter_km(out, R_km=R_km)
    make_postfit_residuals_linear_plot(result, outdir)
    make_trace_cov_pos_vel_plot(result, outdir, length_unit="m")
    make_all_state_3sigma_envelope_plot(out, outdir)


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


def combine_outputs(parts: list[dict]) -> dict:
    if len(parts) == 0:
        raise ValueError("No output parts to combine.")

    out = dict(parts[-1])
    arr_keys = [
        "t_meas",
        "xhat_meas",
        "Xhat_meas",
        "X_pf",
        "P_meas",
        "P_pf",
        "two_sigma_meas",
        "prefit_resids_final",
        "postfit_resids_linear_final",
        "postfit_resids_meas",
        "X_pred_hist",
        "P_pred_hist",
        "Phi_step_hist",
        "H_pred_hist",
        "y_hist",
        "meas_valid",
        "gap_step_mask",
        "snc_suppressed_mask",
        "nis_hist",
        "iter_counts",
    ]

    for k in arr_keys:
        arrs = [np.asarray(p[k], dtype=float) for p in parts if p.get(k, None) is not None]
        if len(arrs) == len(parts) and len(arrs) > 0:
            out[k] = np.concatenate(arrs, axis=0)

    if all(("station_meas" in p) and (p["station_meas"] is not None) for p in parts):
        out["station_meas"] = [s for p in parts for s in p["station_meas"]]

    if any(p.get("state_error_meas", None) is None for p in parts):
        out["state_error_meas"] = None
    else:
        out["state_error_meas"] = np.concatenate(
            [np.asarray(p["state_error_meas"], dtype=float) for p in parts],
            axis=0,
        )

    out["R"] = np.asarray(parts[0].get("R", out.get("R")), dtype=float)
    out["rms_final"] = parts[-1].get("rms_final", out.get("rms_final"))
    out["rms_by_iter"] = parts[-1].get("rms_by_iter", out.get("rms_by_iter"))

    if ("xhat_meas" not in out) and ("Xhat_meas" in out):
        out["xhat_meas"] = out["Xhat_meas"]
    if ("Xhat_meas" not in out) and ("xhat_meas" in out):
        out["Xhat_meas"] = out["xhat_meas"]

    return out


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


def save_final_summary_json(
    out: dict,
    save_path: Path,
    history_path: Path,
    *,
    break_interval_days: float,
    break_days: list[float],
    additional_break_days: list[float],
    stage1_start_day: float,
    inflation_factor: float,
):
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
        "stop_start_settings": {
            "stage1_start_day": float(stage1_start_day),
            "break_interval_days": float(break_interval_days),
            "break_days": [float(d) for d in break_days],
            "additional_break_days": [float(d) for d in additional_break_days],
            "covariance_inflation_factor": float(inflation_factor),
        },
        "iekf_stop_start_6state": {
            "fixed_Cr": float(CR_FIXED),
            "final_state_iekf_forward": x_fwd.tolist(),
            "final_cov_iekf_forward": P_fwd.tolist(),
            "final_cov_iekf_forward_full_km": P_fwd.tolist(),
            "final_time_s_iekf_forward": t_fwd,
        },
        #
        "newekf_6state": {
            "fixed_Cr": float(CR_FIXED),
            "final_state_iekf_forward": x_fwd.tolist(),
            "final_cov_iekf_forward": P_fwd.tolist(),
            "final_cov_iekf_forward_full_km": P_fwd.tolist(),
            "final_time_s_iekf_forward": t_fwd,
        },
      
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
    obs_path = Path(__file__).resolve().parent / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs(obs_path)
    stations = build_stations()

    stage_meas_list, break_days, t_epoch = split_measurements_periodic(
        all_meas,
        stage1_start_day=STAGE1_START_DAY,
        break_interval_days=BREAK_INTERVAL_DAYS,
        additional_break_days=ADDITIONAL_BREAK_DAYS,
    )

    p_const, sc_const, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_iekf = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    sigma_acc_km_s2 = 1.0e-9 # SNC
    Q_iekf = np.diag([sigma_acc_km_s2**2, sigma_acc_km_s2**2, sigma_acc_km_s2**2])

    dyn_iekf = lambda tau, x: mu_sun_srp_state_deriv_for6state(
        t=tau,
        X=x,
        pConst=p_const,
        scConst=sc_const,
        Cr=CR_FIXED,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac_iekf(tau, x):
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

    out_parts: list[dict] = []
    x_seed = np.asarray(x0_iekf, dtype=float).copy()
    P_seed = np.asarray(P0_iekf, dtype=float).copy()
    t_prev_seed = float(t_epoch)

    for i, stage_meas in enumerate(stage_meas_list):
        iekf_stage = IEKF2(
            x0=x_seed,
            P0=P_seed,
            R=R_iekf,
            Q=Q_iekf,
            dyn_fun=dyn_iekf,
            dyn_jac=jac_iekf,
            prop_settings=PropSettings(rtol=1.0e-10, atol=1.0e-10, method="RK45"),
            first_pass_gap_s=6 * 3600.0,
        )
        out_stage = iekf_stage.run(
            all_meas=stage_meas,
            stations=stations,
            Xtrue_meas=None,
            t_prev_init=t_prev_seed,
            iterated=True,
            max_iter=10,
            iter_tol=1.0e-10,
            bound_level=5.0,
            no_snc_on_first_after_gap=True,
            gap_threshold_s=6 * 3600.0,
            show_progress=True,
            progress_every=50,
        )
        out_parts.append(out_stage)

        t_stage = np.asarray(out_stage["t_meas"], dtype=float)
        d0 = float(t_stage[0] / 86400.0)
        d1 = float(t_stage[-1] / 86400.0)
        print(
            f"IEKF Stop-Start Stage {i+1:02d} Complete. "
            f"Days {d0:.3f}-{d1:.3f}, Updates: {len(out_stage['t_meas'])}"
        )

        x_seed = np.asarray(out_stage["xhat_meas"][-1], dtype=float).reshape(-1)
        P_seed = np.asarray(out_stage["P_meas"][-1], dtype=float)
        t_prev_seed = float(t_stage[-1])

        # Covariance bump at each handoff between stages
        if i < (len(stage_meas_list) - 1):
            P_seed = P_seed * float(COVARIANCE_INFLATION_FACTOR)

    out_total = combine_outputs(out_parts)
    print(f"IEKF Stop-Start Total Complete. Number Of Updates: {len(out_total['t_meas'])}")

    plot_root = Path(__file__).resolve().parent / "Plots" / "IEKF Stop Start"
    chunk_root = plot_root / "Chunks"
    total_root = plot_root / "Total Run"
    chunk_root.mkdir(parents=True, exist_ok=True)
    total_root.mkdir(parents=True, exist_ok=True)

    history_path = plot_root / "iekf_stop_start_history_for_end_batch.npz"
    save_history(out_total, history_path)
    final_json_path = plot_root / "final_states_iekf_stop_start.json"
    save_final_summary_json(
        out_total,
        final_json_path,
        history_path,
        break_interval_days=BREAK_INTERVAL_DAYS,
        break_days=break_days,
        additional_break_days=ADDITIONAL_BREAK_DAYS,
        stage1_start_day=STAGE1_START_DAY,
        inflation_factor=COVARIANCE_INFLATION_FACTOR,
    )

    make_standard_plot_set(out_total, total_root, R_km=R_iekf)

    # 50-day chunk plots: [0,50], (50,100], ...
    t_days = np.asarray(out_total["t_meas"], dtype=float) / 86400.0
    max_day = float(np.max(t_days))

    day_start = 0.0
    chunk_i = 1
    while day_start < (max_day - 1.0e-12):
        day_end = min(day_start + 50.0, max_day)
        out_chunk = slice_out_by_day_window(
            out_total,
            day_start,
            day_end,
            include_start=(chunk_i == 1),
        )
        if out_chunk is not None:
            d0 = int(np.floor(day_start + 1.0e-9))
            d1 = int(np.ceil(day_end - 1.0e-9))
            chunk_dir = chunk_root / f"Days_{d0:03d}_{d1:03d}"
            make_standard_plot_set(out_chunk, chunk_dir, R_km=R_iekf)
            print(
                f"Saved Chunk {chunk_i:02d} Plots: Days {day_start:.3f} To {day_end:.3f} "
                f"({len(out_chunk['t_meas'])} measurements)"
            )
        day_start = day_end
        chunk_i += 1

    xf = np.asarray(out_total["xhat_meas"][-1], dtype=float).reshape(-1)
    print("\nFinal State Estimate (IEKF Stop-Start 6-State / IEKF2):")
    print(f"  Position [km]    : [{xf[0]:.6f}, {xf[1]:.6f}, {xf[2]:.6f}]")
    print(f"  Velocity [km/s]  : [{xf[3]:.9f}, {xf[4]:.9f}, {xf[5]:.9f}]")
    print(f"  Fixed Cr [-]     : {CR_FIXED:.9f}")
    print(f"  Break Interval   : {BREAK_INTERVAL_DAYS:.3f} days")
    if len(break_days) > 0:
        print(f"  Break Days       : {', '.join(f'{d:.1f}' for d in break_days)}")
    print(f"  Cov Inflation    : {COVARIANCE_INFLATION_FACTOR:.1f}x")
    print(f"\nSaved IEKF Stop-Start Plots To: {plot_root}")
    print(f"Saved IEKF Stop-Start Final-State JSON To: {final_json_path}")
    print(f"Saved IEKF Stop-Start Time-Step History To: {history_path}")


if __name__ == "__main__":
    main()
