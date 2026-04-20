import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.Functions.batch import batch_estimate_x0
from src.Functions.propagation import PropSettings, propagate_x_phi_step
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_state_deriv
from src.Functions.jacobians import srp_thirdbody_variational_eq
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_state_estimate_3sigma import make_cr_3sigma_plot

x0_batch_cr = np.array(
    [
        -274096770.76544,
        -92859266.4499061,
        -40199493.6677441,
        32.6704564599943,
        -8.93838913761049,
        -3.87881914050316,
        1.0,
    ],
    dtype=float,
)
P0_batch_cr = np.diag(
    [
        100.0**2,
        100.0**2,
        100.0**2,
        0.1**2,
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


def build_dyn_and_jac(pConst, scConst, earth_state_func, sun_state_func):
    dyn = lambda tau, x: mu_sun_srp_state_deriv(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac(tau, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        r_earth, _ = earth_state_func(tau)
        r_sun, _ = sun_state_func(tau)
        return srp_thirdbody_variational_eq(
            r_sc=x[0:3],
            r_earth=r_earth,
            r_sun=r_sun,
            Cr=float(x[6]),
            area=scConst.area,
            mass=scConst.mass,
            mu_earth=pConst.mu_earth,
            mu_i=pConst.mu_sun,
            solar_flux_1au=scConst.solar_flux_1au,
            c=scConst.c,
            AU_m=scConst.AU_m,
        )

    return dyn, jac


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


def batch_info_to_filter_like_out(info: dict, R_km: np.ndarray) -> dict:
    xhat = np.asarray(info["state_hist"], dtype=float)
    P = np.asarray(info["P_hist"], dtype=float)
    two_sigma = 2.0 * np.sqrt(np.maximum(np.diagonal(P, axis1=1, axis2=2), 0.0))

    if "prefit_resids_final" in info:
        prefit = np.asarray(info["prefit_resids_final"], dtype=float)
    else:
        prefit = np.asarray(
            info.get("prefit_residuals", np.full((xhat.shape[0], 2), np.nan)),
            dtype=float,
        )

    if "postfit_resids_linear_final" in info:
        postfit_lin = np.asarray(info["postfit_resids_linear_final"], dtype=float)
    elif "postfit_resids_linear_hist" in info and len(info["postfit_resids_linear_hist"]) > 0:
        postfit_lin = np.asarray(info["postfit_resids_linear_hist"][-1], dtype=float)
    else:
        postfit_lin = np.asarray(
            info.get("postfit_residuals", np.full((xhat.shape[0], 2), np.nan)),
            dtype=float,
        )

    postfit_nl = np.asarray(
        info.get("postfit_residuals", np.full((xhat.shape[0], 2), np.nan)),
        dtype=float,
    )

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


def make_plot_set(out: dict, outdir: Path, *, R_km: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    result = to_plotting_result_6state_from_filter_km(out, R_km=R_km)
    make_postfit_residuals_linear_plot(result, outdir)
    make_cr_3sigma_plot(out, outdir)


def select_meas_day_window(
    all_meas: list[dict],
    day_start: float,
    day_end: float,
    *,
    include_start: bool,
) -> list[dict]:
    t0 = float(day_start) * 86400.0
    t1 = float(day_end) * 86400.0
    if include_start:
        return [m for m in all_meas if (float(m["t"]) >= t0 and float(m["t"]) <= t1)]
    return [m for m in all_meas if (float(m["t"]) > t0 and float(m["t"]) <= t1)]


def propagate_seed_to_time(
    x_seed: np.ndarray,
    P_seed: np.ndarray,
    t_seed: float,
    t_target: float,
    dyn_fun,
    dyn_jac,
    prop_settings: PropSettings,
):
    if float(t_target) <= float(t_seed) + 1.0e-12:
        return x_seed.copy(), P_seed.copy()

    x_t, Phi = propagate_x_phi_step(
        x0=np.asarray(x_seed, dtype=float),
        t0=float(t_seed),
        t1=float(t_target),
        f=dyn_fun,
        A=dyn_jac,
        settings=prop_settings,
    )
    P_t = Phi @ np.asarray(P_seed, dtype=float) @ Phi.T
    P_t = 0.5 * (P_t + P_t.T)
    return x_t, P_t


def concatenate_chunk_outputs(chunk_outs: list[dict], R_km: np.ndarray) -> dict:
    if len(chunk_outs) == 0:
        raise ValueError("No chunk outputs to concatenate.")

    return {
        "t_meas": np.concatenate([np.asarray(o["t_meas"], dtype=float) for o in chunk_outs], axis=0),
        "station_meas": [s for o in chunk_outs for s in list(o["station_meas"])],
        "xhat_meas": np.concatenate([np.asarray(o["xhat_meas"], dtype=float) for o in chunk_outs], axis=0),
        "Xhat_meas": np.concatenate([np.asarray(o["Xhat_meas"], dtype=float) for o in chunk_outs], axis=0),
        "P_meas": np.concatenate([np.asarray(o["P_meas"], dtype=float) for o in chunk_outs], axis=0),
        "two_sigma_meas": np.concatenate([np.asarray(o["two_sigma_meas"], dtype=float) for o in chunk_outs], axis=0),
        "prefit_resids_final": np.concatenate(
            [np.asarray(o["prefit_resids_final"], dtype=float) for o in chunk_outs], axis=0
        ),
        "postfit_resids_linear_final": np.concatenate(
            [np.asarray(o["postfit_resids_linear_final"], dtype=float) for o in chunk_outs], axis=0
        ),
        "postfit_resids_meas": np.concatenate(
            [np.asarray(o["postfit_resids_meas"], dtype=float) for o in chunk_outs], axis=0
        ),
        "state_error_meas": None,
        "R": np.asarray(R_km, dtype=float),
        "rms_final": None,
        "rms_by_iter": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Chunked iterated-batch run with Cr tracking.")
    parser.add_argument("--chunk-days", type=float, default=50.0, help="Chunk length in days.")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum iterated-batch iterations per chunk.")
    parser.add_argument("--tol", type=float, default=1.0e-8, help="Iterated-batch convergence tolerance.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    obs_path = base / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs(obs_path)
    stations = build_stations()

    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()
    dyn_fun, dyn_jac = build_dyn_and_jac(pConst, scConst, earth_state_func, sun_state_func)
    prop_settings = PropSettings(rtol=1.0e-10, atol=1.0e-10, method="RK45")

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_batch = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    plot_root = base / "Plots" / "Batch Cr"
    chunk_root = plot_root / "Chunks"
    total_root = plot_root / "Total Run"
    chunk_root.mkdir(parents=True, exist_ok=True)
    total_root.mkdir(parents=True, exist_ok=True)

    t_all_days = np.array([float(m["t"]) for m in all_meas], dtype=float) / 86400.0
    max_day = float(np.max(t_all_days))

    chunk_outs: list[dict] = []

    t_carry = None
    x_carry = None
    P_carry = None

    day_start = 0.0
    chunk_i = 1
    while day_start < (max_day - 1.0e-12):
        day_end = min(day_start + float(args.chunk_days), max_day)
        meas_chunk = select_meas_day_window(
            all_meas,
            day_start,
            day_end,
            include_start=(chunk_i == 1),
        )
        if len(meas_chunk) == 0:
            day_start = day_end
            chunk_i += 1
            continue

        t_chunk0 = float(meas_chunk[0]["t"])
        if chunk_i == 1:
            x0_bar = x0_batch_cr.copy()
            P0 = P0_batch_cr.copy()
        else:
            x0_bar, P0 = propagate_seed_to_time(
                x_seed=x_carry,
                P_seed=P_carry,
                t_seed=t_carry,
                t_target=t_chunk0,
                dyn_fun=dyn_fun,
                dyn_jac=dyn_jac,
                prop_settings=prop_settings,
            )

        x0_hat, P0_hat, info = batch_estimate_x0(
            all_meas=meas_chunk,
            stations=stations,
            x0_bar=x0_bar,
            P0=P0,
            R=R_batch,
            mu=398600.4415,
            J2=0.0,
            J3=0.0,
            Re=6378.1363,
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            reltol=1.0e-10,
            abstol=1.0e-10,
            method="RK45",
            dyn_fun=dyn_fun,
            dyn_jac=dyn_jac,
            prop_settings=prop_settings,
        )
        out_chunk = batch_info_to_filter_like_out(info, R_km=R_batch)
        chunk_outs.append(out_chunk)

        t_carry = float(out_chunk["t_meas"][-1])
        x_carry = np.asarray(out_chunk["xhat_meas"][-1], dtype=float).copy()
        P_carry = np.asarray(out_chunk["P_meas"][-1], dtype=float).copy()

        d0 = int(np.floor(day_start + 1.0e-9))
        d1 = int(np.ceil(day_end - 1.0e-9))
        chunk_dir = chunk_root / f"Days_{d0:03d}_{d1:03d}"
        make_plot_set(out_chunk, chunk_dir, R_km=R_batch)

        print(
            f"Chunk {chunk_i:02d} Complete: Days {day_start:.3f} -> {day_end:.3f}, "
            f"{len(out_chunk['t_meas'])} measurements, num_iters={info.get('num_iters', 'n/a')}"
        )
        print(f"  Chunk X0 Hat Cr [-]: {float(np.asarray(x0_hat, dtype=float)[6]):.9f}")
        print(f"  Chunk P0 Hat Cr Sigma: {np.sqrt(max(float(np.asarray(P0_hat, dtype=float)[6, 6]), 0.0)):.9e}")

        day_start = day_end
        chunk_i += 1

    out_total = concatenate_chunk_outputs(chunk_outs, R_km=R_batch)
    make_plot_set(out_total, total_root, R_km=R_batch)

    xf = np.asarray(out_total["xhat_meas"][-1], dtype=float)
    print("\nFinal Chunked Iterated-Batch State Estimate:")
    print(f"  Position [km]    : [{xf[0]:.6f}, {xf[1]:.6f}, {xf[2]:.6f}]")
    print(f"  Velocity [km/s]  : [{xf[3]:.9f}, {xf[4]:.9f}, {xf[5]:.9f}]")
    print(f"  Cr [-]           : {xf[6]:.9f}")
    print(f"\nSaved Plots To: {plot_root}")


if __name__ == "__main__":
    main()
