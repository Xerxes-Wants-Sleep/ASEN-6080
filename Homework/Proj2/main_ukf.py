import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append("../../")

from src.Functions.filters import UnscentedKalmanFilter
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_state_deriv
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing_18
from src.helpers.plotting.plot_prefit_residuals import make_prefit_residuals_plot
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_trace_cov import make_trace_cov_plot
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.plot_final_state_estimate import make_final_state_estimate_plot


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


def load_project2_obs_for_ukf(obs_path: Path) -> list[dict]:
    """
    Convert Project2b_Obs.txt to:
      {"station": str, "t": float, "rho_km": float, "rho_dot_km_s": float}
    """
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


def build_problem_constants():
    Jd0 = 2456296.25
    mu_sun = 132712440017.987  # km^3/s^2
    AU_km = 149597870.7
    solar_flux_W_m2 = 1357.0
    SRP_area_mass_ratio = 0.01  # m^2/kg

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


def main():
    obs_path = Path(__file__).resolve().parent / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs_for_ukf(obs_path)
    stations = build_stations()

    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_ukf = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    # Keep your assignment-specific initial condition.
    x0_ukf = np.array(
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
    P0_ukf = np.diag(
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
    # Q_ukf = np.zeros((7, 7), dtype=float)
    # SNC test value
    sigma_acc_km_s2 = 1.0e-11
    Q_ukf = np.diag([sigma_acc_km_s2**2, sigma_acc_km_s2**2, sigma_acc_km_s2**2])

    dyn_ukf = lambda tau, x: mu_sun_srp_state_deriv(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    ukf = UnscentedKalmanFilter(
        X0=x0_ukf,
        P0=P0_ukf,
        R=R_ukf,
        Q=Q_ukf,
        mu=pConst.mu_earth,
        J2=0.0,
        J3=0.0,
        Re=6378.1363,
        alpha=0.8,
        beta=2.0,
        kappa=None,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="RK45",
        j2=False,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
        dyn_fun=dyn_ukf,
    )
    ukf.q_frame = "eci"

    out_ukf = ukf.run(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=None,
        show_progress=True,
        progress_every=50,
    )
    print(f"UKF Run Complete. Number Of Updates: {len(out_ukf['t_meas'])}")

    # Reuse existing plotting pipeline; convert km->m for trace(P) plot labeling.
    result = run_filter_post_processing_18(
        out=out_ukf,
        length_unit_in="km",
        length_unit_out="m",
    )

    outdir = Path(__file__).resolve().parent / "Plots" / "Main UKF with SNC"
    outdir.mkdir(parents=True, exist_ok=True)

    make_prefit_residuals_plot(result, outdir)
    make_postfit_residuals_linear_plot(result, outdir)
    make_trace_cov_plot(result, outdir)
    make_trace_cov_pos_vel_plot(result, outdir, length_unit="m")
    make_final_state_estimate_plot(out_ukf, outdir)

    print(f"Saved Plots To: {outdir}")


if __name__ == "__main__":
    main()
