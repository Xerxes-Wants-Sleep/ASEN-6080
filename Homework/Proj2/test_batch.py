import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.Functions.batch import batch_estimate_x0
from src.Functions.calcB_plane import calc_bplane
from src.Functions.Ephem import ephem
from src.Functions.jacobians import srp_thirdbody_variational_eq
from src.Functions.propagation import PropSettings
from src.Functions.srp_dyn_model import mu_sun_srp_state_deriv, mu_sun_srp_stm_deriv
from src.Functions.SOIcheck import SOIcheck
from src.Functions.stations import Stations


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
    jd0 = 2456296.25
    mu_sun = 132712440017.987  # km^3/s^2
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


def build_dyn_and_jac(p_const, sc_const, earth_state_func, sun_state_func):
    dyn = lambda tau, x: mu_sun_srp_state_deriv(
        t=tau,
        X=x,
        pConst=p_const,
        scConst=sc_const,
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
            area=sc_const.area,
            mass=sc_const.mass,
            mu_earth=p_const.mu_earth,
            mu_i=p_const.mu_sun,
            solar_flux_1au=sc_const.solar_flux_1au,
            c=sc_const.c,
            AU_m=sc_const.AU_m,
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


def propagate_snapshot_to_3soi_with_stm(
    *,
    x_arc: np.ndarray,
    P_arc: np.ndarray,
    t_arc: float,
    p_const,
    sc_const,
    earth_state_func,
    sun_state_func,
    t_search_days: float = 400.0,
):
    y0 = np.hstack((np.asarray(x_arc, dtype=float).reshape(7), np.eye(7).reshape(-1)))

    soi_event = lambda tau, y: SOIcheck(tau, y[:7])
    soi_event.terminal = True
    soi_event.direction = -1.0

    sol = solve_ivp(
        fun=lambda tau, y: mu_sun_srp_stm_deriv(
            t=tau,
            XPhi=y,
            pConst=p_const,
            scConst=sc_const,
            earth_state_func=earth_state_func,
            sun_state_func=sun_state_func,
        ),
        t_span=(float(t_arc), float(t_arc) + float(t_search_days) * 86400.0),
        y0=y0,
        events=soi_event,
        rtol=1.0e-12,
        atol=1.0e-12,
        method="RK45",
        max_step=3600.0,
    )

    if (not sol.success) or (len(sol.t_events[0]) == 0):
        raise RuntimeError(f"3*RSOI crossing not found from arc at t={t_arc:.3f} s")

    t_3soi = float(sol.t_events[0][0])
    y_3soi = np.asarray(sol.y_events[0][0], dtype=float)
    x_3soi = y_3soi[:7]
    phi_arc_to_3soi = y_3soi[7:].reshape(7, 7)
    P_3soi = phi_arc_to_3soi @ np.asarray(P_arc, dtype=float) @ phi_arc_to_3soi.T
    return t_3soi, x_3soi, P_3soi


def main():
    base = Path(__file__).resolve().parent

    obs_path = base / "Given_data" / "Project2a_Obs.txt"
    all_meas = load_project2_obs(obs_path)
    stations = build_stations()

    p_const, sc_const, earth_state_func, sun_state_func = build_problem_constants()
    dyn_fun, dyn_jac = build_dyn_and_jac(p_const, sc_const, earth_state_func, sun_state_func)

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_batch = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    x0 = np.array(
        [-274096790.0, -92859240.0, -40199490.0, 32.67, -8.94, -3.88, 1.2],
        dtype=float,
    )
    P0 = np.diag([100.0**2, 100.0**2, 100.0**2, 0.1**2, 0.1**2, 0.1**2, 0.1**2]).astype(float)

    with contextlib.redirect_stdout(io.StringIO()):
        _x0_hat, _P0_hat, info = batch_estimate_x0(
            all_meas=all_meas,
            stations=stations,
            x0_bar=x0,
            P0=P0,
            R=R_batch,
            mu=398600.4415,
            J2=0.0,
            J3=0.0,
            Re=6378.1363,
            max_iter=10,
            tol=1.0e-8,
            reltol=1.0e-10,
            abstol=1.0e-10,
            method="RK45",
            dyn_fun=dyn_fun,
            dyn_jac=dyn_jac,
            prop_settings=PropSettings(rtol=1.0e-10, atol=1.0e-10, method="RK45"),
        )

    t_meas = np.asarray(info["t_meas"], dtype=float)
    xhat = np.asarray(info["state_hist"], dtype=float)
    P_hist = np.asarray(info["P_hist"], dtype=float)
    t_days = t_meas / 86400.0

    arc_days_target = [50.0, 100.0, 150.0, 200.0]

    for d_target in arc_days_target:
        idx = int(np.argmin(np.abs(t_days - float(d_target))))
        x_arc = xhat[idx, :]
        P_arc = P_hist[idx, :, :]
        t_arc = float(t_meas[idx])
        d_sel = float(t_days[idx])

        t_3soi_arc, x_3soi_arc, P_3soi_arc = propagate_snapshot_to_3soi_with_stm(
            x_arc=x_arc,
            P_arc=P_arc,
            t_arc=t_arc,
            p_const=p_const,
            sc_const=sc_const,
            earth_state_func=earth_state_func,
            sun_state_func=sun_state_func,
            t_search_days=400.0,
        )

        (
            bdot_r_arc,
            bdot_t_arc,
            sig_r_arc,
            sig_t_arc,
            _sig_rt_arc,
            _x_cross,
            _p_bplane_arc,
            _str2eci,
            _xphi_bplane,
            _t_bplane,
        ) = calc_bplane(
            XPhi_3SOI=x_3soi_arc,
            t_3SOI=t_3soi_arc,
            P_3SOI=P_3soi_arc,
            pConst=p_const,
            scConst=sc_const,
            earth_state_func=earth_state_func,
            sun_state_func=sun_state_func,
        )

        print(
            f"Arc Target {d_target:6.1f} Days (Selected {d_sel:7.2f}): "
            f"BdotT={float(bdot_t_arc):10.3f} km, "
            f"BdotR={float(bdot_r_arc):10.3f} km, "
            f"sigmaT={float(sig_t_arc):10.3f} km, "
            f"sigmaR={float(sig_r_arc):10.3f} km"
        )


if __name__ == "__main__":
    main()
