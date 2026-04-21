import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.Functions.batch_18_state import batch_estimate_x0
from src.Functions.propagation import PropSettings
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model_6_state import mu_sun_srp_state_deriv_for6state
from src.Functions.jacobians import srp_thirdbody_variational_eq_for6state
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot


CR_FIXED = 1.38
RUN_DAYS = 10.0

x0_batch_start = np.array(
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
P0_batch_start = np.diag(
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


def select_first_days(all_meas: list[dict], days: float) -> list[dict]:
    t_end = float(days) * 86400.0
    meas = [m for m in all_meas if float(m["t"]) <= (t_end + 1.0e-12)]
    if len(meas) < 2:
        raise ValueError(f"Window [0,{days}] days has too few measurements.")
    return meas


def batch_info_to_filter_like_out(info: dict, R_km: np.ndarray) -> dict:
    xhat = np.asarray(info["state_hist"], dtype=float)
    P = np.asarray(info["P_hist"], dtype=float)
    two_sigma = 2.0 * np.sqrt(np.maximum(np.diagonal(P, axis1=1, axis2=2), 0.0))

    postfit_lin = None
    if "postfit_resids_linear_hist" in info and len(info["postfit_resids_linear_hist"]) > 0:
        postfit_lin = np.asarray(info["postfit_resids_linear_hist"][-1], dtype=float)

    out = {
        "t_meas": np.asarray(info["t_meas"], dtype=float),
        "station_meas": list(info["station_meas"]),
        "xhat_meas": xhat,
        "Xhat_meas": xhat,
        "P_meas": P,
        "two_sigma_meas": two_sigma,
        "state_error_meas": None,
        "prefit_resids_final": np.asarray(info["prefit_residuals"], dtype=float),
        "postfit_resids_linear_final": postfit_lin
        if postfit_lin is not None
        else np.asarray(info["postfit_residuals"], dtype=float),
        "postfit_resids_meas": np.asarray(info["postfit_residuals"], dtype=float),
        "rms_final": None,
        "rms_by_iter": None,
        "R": np.asarray(R_km, dtype=float),
    }
    return out


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


def make_new_x0_plot(x0_initial: np.ndarray, x0_new: np.ndarray, outdir: Path):
    labels = ["x", "y", "z", "vx", "vy", "vz"]
    idx = np.arange(6)
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(idx - width / 2, x0_initial, width=width, label="Initial x0")
    ax.bar(idx + width / 2, x0_new, width=width, label="Estimated new x0")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_title("Initial Vs Estimated x0 (6-State Batch Start)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "new_x0_comparison.png", dpi=150)
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    obs_path = base / "Given_data" / "Project2b_Obs.txt"
    outdir = base / "Plots" / "Batch Start"
    outdir.mkdir(parents=True, exist_ok=True)

    all_meas = load_project2_obs(obs_path)
    meas_start = select_first_days(all_meas, RUN_DAYS)
    stations = build_stations()
    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_batch = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    dyn_fun = lambda tau, x: mu_sun_srp_state_deriv_for6state(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        Cr=CR_FIXED,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def dyn_jac(tau, x):
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

    x0_hat, P0_hat, info = batch_estimate_x0(
        all_meas=meas_start,
        stations=stations,
        x0_bar=x0_batch_start,
        P0=P0_batch_start,
        R=R_batch,
        dyn_fun=dyn_fun,
        dyn_jac=dyn_jac,
        max_iter=10,
        tol=1.0e-8,
        prop_settings=PropSettings(rtol=1.0e-10, atol=1.0e-10, method="RK45"),
    )

    out = batch_info_to_filter_like_out(info, R_km=R_batch)
    postfit_plot_out = to_plotting_result_6state_from_filter_km(out, R_km=R_batch)
    make_postfit_residuals_linear_plot(postfit_plot_out, outdir)
    make_new_x0_plot(x0_batch_start, x0_hat, outdir)

    x0_csv = outdir / "new_x0_batch_start.csv"
    pd.DataFrame(
        {
            "component": ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"],
            "x0_initial": x0_batch_start,
            "x0_estimated": np.asarray(x0_hat, dtype=float).reshape(-1),
        }
    ).to_csv(x0_csv, index=False)

    x0_json = outdir / "new_x0_batch_start.json"
    x0_payload = {
        "units": {
            "position": "km",
            "velocity": "km/s",
            "covariance": "km^2, km^2/s^2, cross terms consistent with state units",
        },
        "x0_initial": np.asarray(x0_batch_start, dtype=float).tolist(),
        "x0_estimated": np.asarray(x0_hat, dtype=float).reshape(-1).tolist(),
        "P0_estimated": np.asarray(P0_hat, dtype=float).tolist(),
        "meta": {
            "run_days": float(RUN_DAYS),
            "cr_fixed": float(CR_FIXED),
            "num_measurements": int(len(meas_start)),
            "num_iters": int(info.get("num_iters", 0)),
        },
    }
    with open(x0_json, "w", encoding="utf-8") as f:
        json.dump(x0_payload, f, indent=2)

    print("\nBatch Start (6-State) Complete.")
    print(f"  Data Arc: 0.0 To {RUN_DAYS:.1f} Days")
    print(f"  Measurements Used: {len(meas_start)}")
    print(f"  Batch Iterations: {info.get('num_iters', 'n/a')}")
    print("\nEstimated New x0 [km, km/s]:")
    print(np.asarray(x0_hat, dtype=float).reshape(-1))
    print("\nSaved:")
    print(f"  Postfit Plot: {outdir / 'postfit_residuals_linear.png'}")
    print(f"  New x0 Plot : {outdir / 'new_x0_comparison.png'}")
    print(f"  New x0 CSV  : {x0_csv}")
    print(f"  New x0 JSON : {x0_json}")
    print(f"  New P0 shape: {np.asarray(P0_hat).shape}")


if __name__ == "__main__":
    main()
