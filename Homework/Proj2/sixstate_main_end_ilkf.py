import argparse
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.Functions.filter_18_state import LinearizedKalmanFilter18State
from src.Functions.propagation import PropSettings
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model_6_state import mu_sun_srp_state_deriv_for6state
from src.Functions.jacobians import srp_thirdbody_variational_eq_for6state
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_state_estimate_3sigma import make_all_state_3sigma_envelope_plot


CR_FIXED_DEFAULT = 1.38


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
    Jd0 = 2456296.25
    mu_sun = 132712440017.987
    AU_km = 149597870.7
    solar_flux_W_m2 = 1357.0
    SRP_area_mass_ratio = 0.01

    pConst = type("pConst", (), {})()
    pConst.mu_earth = 398600.4415
    pConst.mu_sun = mu_sun

    scConst = type("scConst", (), {})()
    scConst.area = SRP_area_mass_ratio
    scConst.mass = 1.0
    scConst.solar_flux_1au = solar_flux_W_m2
    scConst.c = 299792458.0
    scConst.AU_m = AU_km * 1000.0

    def earth_state_func(tau):
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    def sun_state_func(tau):
        jd = Jd0 + float(tau) / 86400.0
        rE_km, vE_km_s, _ = ephem(jd, 3, frame="EME2000")
        rE_km = np.asarray(rE_km, dtype=float).reshape(3,)
        vE_km_s = np.asarray(vE_km_s, dtype=float).reshape(3,)
        return -rE_km, -vE_km_s

    return pConst, scConst, earth_state_func, sun_state_func


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


def load_maneuver_iekf_histories(history_path: Path) -> dict:
    if not history_path.exists():
        raise FileNotFoundError(
            f"Missing maneuver IEKF history file: {history_path}\n"
            "Run main_maneuver_check.py first so it saves forward history."
        )

    data = np.load(history_path)
    return {
        "t_meas": np.asarray(data["t_meas"], dtype=float),
        "forward": {
            "xhat_meas": np.asarray(data["xhat_meas"], dtype=float),
            "P_meas": np.asarray(data["P_meas"], dtype=float),
        },
    }


def select_tail_measurements(all_meas: list[dict], tail_days: float):
    t_all = np.array([float(m["t"]) for m in all_meas], dtype=float)
    if t_all.size == 0:
        raise ValueError("No measurements found in observation file.")
    t_end = float(np.max(t_all))
    t_start = max(0.0, t_end - float(tail_days) * 86400.0)
    tail = [m for m in all_meas if float(m["t"]) >= (t_start - 1.0e-12)]
    if len(tail) == 0:
        raise ValueError("Tail selection produced no measurements.")
    return tail, t_start, t_end


def seed_from_iekf_history(history: dict, t_seed: float):
    t_hist = np.asarray(history["t_meas"], dtype=float)
    x_hist = np.asarray(history["xhat_meas"], dtype=float)
    P_hist = np.asarray(history["P_meas"], dtype=float)

    idx = int(np.searchsorted(t_hist, t_seed - 1.0e-12, side="left"))
    if idx >= t_hist.size:
        idx = t_hist.size - 1

    if not np.isclose(t_hist[idx], t_seed, rtol=0.0, atol=1.0e-7):
        idx_near = int(np.argmin(np.abs(t_hist - t_seed)))
        if abs(float(t_hist[idx_near]) - float(t_seed)) > 1.0:
            raise ValueError(
                "Could not align IEKF history with tail-arc start time. "
                "Re-run sixstate_main_newekf.py on the same obs file."
            )
        idx = idx_near

    x0_star = np.asarray(x_hist[idx], dtype=float).copy()
    P0 = np.asarray(P_hist[idx], dtype=float).copy()
    if x0_star.shape[0] < 6:
        raise ValueError(f"Expected at least 6 states in seed history, got shape {x0_star.shape}.")
    if P0.shape[0] < 6 or P0.shape[1] < 6:
        raise ValueError(f"Expected covariance with at least 6x6 block, got shape {P0.shape}.")

    x0_star_6 = np.asarray(x0_star[:6], dtype=float).copy()
    P0_6 = np.asarray(P0[:6, :6], dtype=float).copy()

    if not np.all(np.isfinite(x0_star_6)) or not np.all(np.isfinite(P0_6)):
        raise ValueError("Seed state/covariance from IEKF history contains non-finite values.")
    return x0_star_6, P0_6, idx, float(t_hist[idx])


def make_lkf_filter(
    x0_star: np.ndarray,
    P0: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    pConst,
    scConst,
    earth_state_func,
    sun_state_func,
    cr_fixed: float,
):
    dyn_lkf = lambda tau, x: mu_sun_srp_state_deriv_for6state(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        Cr=cr_fixed,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac_lkf(tau, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        r_earth, _ = earth_state_func(tau)
        r_sun, _ = sun_state_func(tau)
        return srp_thirdbody_variational_eq_for6state(
            r_sc=x[0:3],
            r_earth=r_earth,
            r_sun=r_sun,
            Cr=cr_fixed,
            area=scConst.area,
            mass=scConst.mass,
            mu_earth=pConst.mu_earth,
            mu_i=pConst.mu_sun,
            solar_flux_1au=scConst.solar_flux_1au,
            c=scConst.c,
            AU_m=scConst.AU_m,
        )

    return LinearizedKalmanFilter18State(
        X0_star=x0_star,
        P0=P0,
        R=R,
        Q=Q,
        dyn_fun=dyn_lkf,
        dyn_jac=jac_lkf,
        prop_settings=PropSettings(rtol=1.0e-10, atol=1.0e-10, method="RK45"),
        first_pass_gap_s=6 * 3600.0,
    )


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


def make_tail_plot_set(out: dict, outdir: Path, *, R_km: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    postfit_plot_out = to_plotting_result_6state_from_filter_km(out, R_km=R_km)
    make_postfit_residuals_linear_plot(postfit_plot_out, outdir)
    make_all_state_3sigma_envelope_plot(out, outdir)


def make_smoothed_plot_out(forward_out: dict) -> dict | None:
    if ("Xhat_smooth" not in forward_out) or ("P_smooth" not in forward_out):
        return None

    out_s = dict(forward_out)
    out_s["xhat_meas"] = np.asarray(forward_out["Xhat_smooth"], dtype=float)
    out_s["Xhat_meas"] = np.asarray(forward_out["Xhat_smooth"], dtype=float)
    out_s["P_meas"] = np.asarray(forward_out["P_smooth"], dtype=float)
    out_s["two_sigma_meas"] = np.asarray(
        forward_out.get("two_sigma_smooth", np.full_like(out_s["xhat_meas"], np.nan)),
        dtype=float,
    )

    if forward_out.get("state_error_smooth_meas", None) is not None:
        out_s["state_error_meas"] = np.asarray(forward_out["state_error_smooth_meas"], dtype=float)
    elif forward_out.get("state_error_smooth", None) is not None:
        out_s["state_error_meas"] = np.asarray(forward_out["state_error_smooth"], dtype=float)
    else:
        out_s["state_error_meas"] = None

    postfit_lin = forward_out.get("postfit_resids_linear_smooth", None)
    if postfit_lin is not None:
        out_s["postfit_resids_linear_final"] = np.asarray(postfit_lin, dtype=float)
        out_s["postfit_resids_meas"] = np.asarray(postfit_lin, dtype=float)

    return out_s


def main():
    parser = argparse.ArgumentParser(
        description="Run six-state end-arc ILKF using Maneuver IEKF forward seed at end-minus-tail-days."
    )
    parser.add_argument(
        "--tail-days",
        type=float,
        default=20.0,
        help="Number of trailing days to run ILKF over (default: 20).",
    )
    parser.add_argument("--max-iters", type=int, default=10, help="Maximum outer ILKF iterations.")
    parser.add_argument("--iter-tol", type=float, default=1.0e-10, help="Outer ILKF convergence tolerance on ||dX0||.")
    parser.add_argument(
        "--snc-accel",
        type=float,
        default=1.0e-11,
        help="SNC acceleration sigma [km/s^2]. Use 0.0 for no SNC.",
    )
    parser.add_argument(
        "--cr-fixed",
        type=float,
        default=CR_FIXED_DEFAULT,
        help="Fixed Cr used in 6-state dynamics (not estimated).",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "Plots" / "Maneuver Check IEKF" / "maneuver_check_history.npz"),
        help="Path to saved maneuver IEKF history from main_maneuver_check.py",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    obs_path = base / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs(obs_path)

    tail_meas, t_start, t_end = select_tail_measurements(all_meas, tail_days=args.tail_days)
    t_seed = float(tail_meas[0]["t"])

    histories = load_maneuver_iekf_histories(Path(args.history_path))

    stations = build_stations()
    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_lkf = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    sigma_acc = float(args.snc_accel)
    if sigma_acc <= 0.0:
        Q_lkf = np.zeros((6, 6), dtype=float)
    else:
        Q_lkf = np.diag([sigma_acc**2, sigma_acc**2, sigma_acc**2])

    def run_case(seed_label: str, history_seed: dict):
        x0_star, P0, idx_seed, t_seed_hist = seed_from_iekf_history(history_seed, t_seed=t_seed)

        lkf = make_lkf_filter(
            x0_star=x0_star,
            P0=P0,
            R=R_lkf,
            Q=Q_lkf,
            pConst=pConst,
            scConst=scConst,
            earth_state_func=earth_state_func,
            sun_state_func=sun_state_func,
            cr_fixed=float(args.cr_fixed),
        )

        out = lkf.run_iterated(
            all_meas=tail_meas,
            stations=stations,
            Xtrue_meas=None,
            max_iters=int(args.max_iters),
            iter_tol=float(args.iter_tol),
            show_progress=False,
            progress_every=50,
            verbose=True,
        )

        if hasattr(lkf, "smooth"):
            lkf.smooth(run_out=out, stations=stations)
        out_smooth = make_smoothed_plot_out(out)

        day0 = float(out["t_meas"][0]) / 86400.0
        day1 = float(out["t_meas"][-1]) / 86400.0
        outdir = (
            base
            / "Plots"
            / "End ILKF Sixstate"
            / f"{seed_label}"
            / f"Days_{int(np.floor(day0)):03d}_{int(np.ceil(day1)):03d}"
        )
        outdir_smooth = outdir / "Smoothed"
        make_tail_plot_set(out, outdir, R_km=R_lkf)
        if out_smooth is not None:
            make_tail_plot_set(out_smooth, outdir_smooth, R_km=R_lkf)

        xf = np.asarray(out["xhat_meas"][-1], dtype=float).reshape(-1)
        Pf = np.asarray(out["P_meas"][-1], dtype=float)
        tf = float(np.asarray(out["t_meas"], dtype=float)[-1])
        xfs = None
        Pfs = None
        tfs = None
        if out_smooth is not None:
            xfs = np.asarray(out_smooth["xhat_meas"][-1], dtype=float).reshape(-1)
            Pfs = np.asarray(out_smooth["P_meas"][-1], dtype=float)
            tfs = float(np.asarray(out_smooth["t_meas"], dtype=float)[-1])

        print(f"\nEnd-Arc Six-State ILKF Complete ({seed_label}).")
        print(f"  Tail Days Requested          : {args.tail_days:.3f}")
        print(f"  Tail Time Window [days]      : {t_start/86400.0:.6f} -> {t_end/86400.0:.6f}")
        print(f"  Seed Measurement Time [days] : {t_seed/86400.0:.6f}")
        print(f"  Seed Matched History Index   : {idx_seed}")
        print(f"  Seed Matched Time [days]     : {t_seed_hist/86400.0:.6f}")
        print(f"  ILKF Iterations Used         : {out.get('iter_count', 'n/a')}")
        print(f"  ILKF Converged               : {out.get('iter_converged', 'n/a')}")
        print(f"\nFinal State Estimate ({seed_label} seed, End-Arc Six-State ILKF):")
        print(f"  Position [km]    : [{xf[0]:.6f}, {xf[1]:.6f}, {xf[2]:.6f}]")
        print(f"  Velocity [km/s]  : [{xf[3]:.9f}, {xf[4]:.9f}, {xf[5]:.9f}]")
        print(f"  Fixed Cr [-]     : {float(args.cr_fixed):.9f}")
        if xfs is not None:
            print(f"\nFinal State Estimate (Smoothed {seed_label} seed End-Arc Six-State ILKF):")
            print(f"  Position [km]    : [{xfs[0]:.6f}, {xfs[1]:.6f}, {xfs[2]:.6f}]")
            print(f"  Velocity [km/s]  : [{xfs[3]:.9f}, {xfs[4]:.9f}, {xfs[5]:.9f}]")
            print(f"  Fixed Cr [-]     : {float(args.cr_fixed):.9f}")
        print(f"\nSaved End-Arc Six-State ILKF Plots To: {outdir}")
        if out_smooth is not None:
            print(f"Saved Smoothed End-Arc Six-State ILKF Plots To: {outdir_smooth}")

        return {
            "out": out,
            "out_smooth": out_smooth,
            "final_forward": xf,
            "final_cov_forward": Pf,
            "final_time_s_forward": tf,
            "final_smoothed": xfs,
            "final_cov_smoothed": Pfs,
            "final_time_s_smoothed": tfs,
            "plot_dir": outdir,
        }

    # Run with forward seed from maneuver IEKF history at t_end - tail_days.
    hist_forward = {
        "t_meas": histories["t_meas"],
        "xhat_meas": histories["forward"]["xhat_meas"],
        "P_meas": histories["forward"]["P_meas"],
    }
    case_forward = run_case("Seed_ForwardIEKF", hist_forward)

    final_json = base / "Plots" / "End ILKF Sixstate" / "final_states_maneuver_seed_comparison.json"
    final_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "history_source": str(Path(args.history_path).resolve()),
        "tail_days": float(args.tail_days),
        "seed_strategy": "forward_iekf_at_end_minus_tail_days",
        "units": {
            "state": "[km, km, km, km/s, km/s, km/s]",
            "covariance": "km-based full-state covariance; position variances in km^2, velocity variances in (km/s)^2, and cross terms in consistent mixed units",
            "time": "seconds since epoch",
        },
        "seed_forward_iekf": {
            "final_state_ilkf_forward": case_forward["final_forward"].tolist(),
            "final_cov_ilkf_forward": case_forward["final_cov_forward"].tolist(),
            "final_cov_ilkf_forward_full_km": case_forward["final_cov_forward"].tolist(),
            "final_time_s_ilkf_forward": case_forward["final_time_s_forward"],
            "final_state_ilkf_smoothed": None if case_forward["final_smoothed"] is None else case_forward["final_smoothed"].tolist(),
            "final_cov_ilkf_smoothed": None if case_forward["final_cov_smoothed"] is None else case_forward["final_cov_smoothed"].tolist(),
            "final_cov_ilkf_smoothed_full_km": None if case_forward["final_cov_smoothed"] is None else case_forward["final_cov_smoothed"].tolist(),
            "final_time_s_ilkf_smoothed": case_forward["final_time_s_smoothed"],
        },
    }
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved labeled final-state comparison JSON to: {final_json}")
    print(f"History Seed Source: {Path(args.history_path).resolve()}")


if __name__ == "__main__":
    main()
