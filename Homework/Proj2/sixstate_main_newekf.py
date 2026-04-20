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


x0_sixstate_newekf = np.array(
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
P0_sixstate_newekf = np.diag(
    [
        100.0**2,
        100.0**2,
        100.0**2,
        0.1**2,
        0.1**2,
        0.1**2,
    ],
)
CR_FIXED = 1.38


def build_problem_constants():
    Jd0 = 2456296.25
    mu_sun = 132712440017.987
    AU_km = 149597870.7
    solar_flux_W_m2 = 1357.0
    SRP_area_mass_ratio = 0.01

    pConst = type("pConst", (), {})()
    pConst.mu_earth = 3.98600432896939e5
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


def make_smoothed_plot_out(forward_out: dict) -> dict | None:
    if ("Xhat_smooth" not in forward_out) or ("P_smooth" not in forward_out):
        return None

    out_s = dict(forward_out)
    out_s["xhat_meas"] = np.asarray(forward_out["Xhat_smooth"], dtype=float)
    out_s["Xhat_meas"] = np.asarray(forward_out["Xhat_smooth"], dtype=float)
    out_s["P_meas"] = np.asarray(forward_out["P_smooth"], dtype=float)
    out_s["P_pf"] = out_s["P_meas"].reshape(out_s["P_meas"].shape[0], -1, order="F")
    out_s["two_sigma_meas"] = np.asarray(
        forward_out.get("two_sigma_smooth", np.full_like(out_s["xhat_meas"], np.nan)),
        dtype=float,
    )
    out_s["state_error_meas"] = forward_out.get("state_error_smooth_meas", None)
    out_s["postfit_resids_linear_final"] = np.asarray(
        forward_out.get("postfit_resids_linear_smooth", np.full((out_s["xhat_meas"].shape[0], 2), np.nan)),
        dtype=float,
    )
    out_s["postfit_resids_meas"] = np.asarray(
        forward_out.get("postfit_resids_smooth_nl", out_s["postfit_resids_linear_final"]),
        dtype=float,
    )
    return out_s


def make_smoothed_plot_set(out_smooth: dict, outdir: Path, *, R_km: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    result = to_plotting_result_6state_from_filter_km(out_smooth, R_km=R_km)
    make_postfit_residuals_linear_plot(result, outdir)
    make_all_state_3sigma_envelope_plot(out_smooth, outdir)


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


def main():
    obs_path = Path(__file__).resolve().parent / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs(obs_path)
    stations = build_stations()

    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_sixstate = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    sigma_acc_km_s2 = 1.0e-8
    Q_sixstate = np.diag([sigma_acc_km_s2**2, sigma_acc_km_s2**2, sigma_acc_km_s2**2])

    dyn_sixstate = lambda tau, x: mu_sun_srp_state_deriv_for6state(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        Cr=CR_FIXED,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac_sixstate(tau, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        r_earth, _ = earth_state_func(tau)
        r_sun, _ = sun_state_func(tau)
        return srp_thirdbody_variational_eq_for6state(
            r_sc=x[0:3],
            r_earth=r_earth,
            r_sun=r_sun,
            Cr=CR_FIXED,
            area=scConst.area,
            mass=scConst.mass,
            mu_earth=pConst.mu_earth,
            mu_i=pConst.mu_sun,
            solar_flux_1au=scConst.solar_flux_1au,
            c=scConst.c,
            AU_m=scConst.AU_m,
        )

    sixstate_newekf = IEKF2(
        x0=x0_sixstate_newekf,
        P0=P0_sixstate_newekf,
        R=R_sixstate,
        Q=Q_sixstate,
        dyn_fun=dyn_sixstate,
        dyn_jac=jac_sixstate,
        prop_settings=PropSettings(rtol=1.0e-10, atol=1.0e-10, method="RK45"),
        first_pass_gap_s=6 * 3600.0,
    )

    out = sixstate_newekf.run(
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
    )
    print(f"Six-State New EKF (IEKF2, Fixed Cr={CR_FIXED:.3f}) Run Complete. Number Of Updates: {len(out['t_meas'])}")
    sixstate_newekf.smooth(out, stations=stations)
    out_smooth = make_smoothed_plot_out(out)

    plot_root = Path(__file__).resolve().parent / "Plots" / "Sixstate New EKF"
    chunk_root = plot_root / "Chunks"
    total_root = plot_root / "Total Run"
    smooth_root = plot_root / "Smoothed Run"
    chunk_root.mkdir(parents=True, exist_ok=True)
    total_root.mkdir(parents=True, exist_ok=True)
    smooth_root.mkdir(parents=True, exist_ok=True)

    history_path = plot_root / "sixstate_newekf_history.npz"
    save_history(out, history_path)

    make_standard_plot_set(out, total_root, R_km=R_sixstate)
    if out_smooth is not None:
        make_smoothed_plot_set(out_smooth, smooth_root, R_km=R_sixstate)

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
            make_standard_plot_set(out_chunk, chunk_dir, R_km=R_sixstate)
            print(
                f"Saved Chunk {chunk_i:02d} Plots: Days {day_start:.3f} To {day_end:.3f} "
                f"({len(out_chunk['t_meas'])} measurements)"
            )
        day_start = day_end
        chunk_i += 1

    xf = np.asarray(out["xhat_meas"][-1], dtype=float).reshape(-1)
    print("\nFinal State Estimate (Six-State New EKF / IEKF2):")
    print(f"  Position [km]    : [{xf[0]:.6f}, {xf[1]:.6f}, {xf[2]:.6f}]")
    print(f"  Velocity [km/s]  : [{xf[3]:.9f}, {xf[4]:.9f}, {xf[5]:.9f}]")
    print(f"  Fixed Cr [-]     : {CR_FIXED:.9f}")
    if out_smooth is not None:
        xfs = np.asarray(out_smooth["xhat_meas"][-1], dtype=float).reshape(-1)
        print("\nFinal State Estimate (Smoothed Six-State New EKF / IEKF2):")
        print(f"  Position [km]    : [{xfs[0]:.6f}, {xfs[1]:.6f}, {xfs[2]:.6f}]")
        print(f"  Velocity [km/s]  : [{xfs[3]:.9f}, {xfs[4]:.9f}, {xfs[5]:.9f}]")
        print(f"  Fixed Cr [-]     : {CR_FIXED:.9f}")
    print(f"\nSaved Six-State New EKF Plots To: {plot_root}")
    print(f"Saved Six-State New EKF Time-Step History To: {history_path}")


if __name__ == "__main__":
    main()
