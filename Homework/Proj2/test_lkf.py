import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append("../../")

from src.Functions.filters import LinearizedKalmanFilter
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_state_deriv
from src.Functions.jacobians import srp_thirdbody_variational_eq
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
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
            lon_deg=-355.749444,
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


def make_postfit_plot(out: dict, outdir: Path, *, R_km: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    result = to_plotting_result_6state_from_filter_km(out, R_km=R_km)
    make_postfit_residuals_linear_plot(result, outdir)


def make_state_estimate_plot(out: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    make_all_state_3sigma_envelope_plot(out, outdir)


def make_lkf_filter(
    x0: np.ndarray,
    P0: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    pConst,
    scConst,
    earth_state_func,
    sun_state_func,
):
    dyn_lkf = lambda tau, x: mu_sun_srp_state_deriv(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac_lkf(tau, x):
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

    return LinearizedKalmanFilter(
        X0_star=x0,
        P0=P0,
        R=R,
        Q=Q,
        mu=pConst.mu_earth,
        J2=0.0,
        J3=0.0,
        Re=6378.1363,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="RK45",
        j2=False,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
        dyn_fun=dyn_lkf,
        dyn_jac=jac_lkf,
    )


def main():
    base = Path(__file__).resolve().parent

    obs_path = base / "Given_data" / "Project2a_Obs.txt"
    all_meas = load_project2_obs(obs_path)
    stations = build_stations()

    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_lkf = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    x0 = np.array(
        [-274096790.0, -92859240.0, -40199490.0, 32.67, -8.94, -3.88, 1.2],
        dtype=float,
    )
    P0 = np.diag([100.0**2, 100.0**2, 100.0**2, 0.1**2, 0.1**2, 0.1**2, 0.1**2])
    Q = np.zeros((7, 7), dtype=float)
    # sigma_acc_km_s2 = 1.0e-10 #SNC
    # Q = np.diag([sigma_acc_km_s2**2, sigma_acc_km_s2**2, sigma_acc_km_s2**2])


    lkf = make_lkf_filter(
        x0=x0,
        P0=P0,
        R=R_lkf,
        Q=Q,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    out = lkf.run(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=None,
        show_progress=True,
        progress_every=50,
    )
    print(f"LKF Run Complete. Number Of Updates: {len(out['t_meas'])}")

    out_root = base / "Validation Plots" / "Test LKF"
    out_root.mkdir(parents=True, exist_ok=True)

    make_postfit_plot(out, out_root, R_km=R_lkf)
    make_state_estimate_plot(out, out_root)

    xf = np.asarray(out["xhat_meas"][-1], dtype=float)
    print("\nFinal State Estimate (LKF):")
    print(f"  Position [km]    : [{xf[0]:.6f}, {xf[1]:.6f}, {xf[2]:.6f}]")
    print(f"  Velocity [km/s]  : [{xf[3]:.9f}, {xf[4]:.9f}, {xf[5]:.9f}]")
    print(f"  Cr [-]           : {xf[6]:.9f}")
    print(f"\nSaved LKF Test Plots To: {out_root}")


if __name__ == "__main__":
    main()
