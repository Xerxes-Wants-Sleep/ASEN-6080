import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append("../../")

from src.Functions.filters import ExtendedKalmanFilter
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_state_deriv
from src.Functions.jacobians import srp_thirdbody_variational_eq
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing_18
from src.helpers.plotting.plot_prefit_residuals import make_prefit_residuals_plot
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_trace_cov import make_trace_cov_plot
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.plot_final_state_estimate import make_final_state_estimate_plot


def print_final_state_summary(label: str, x_final: np.ndarray):
    x_final = np.asarray(x_final, dtype=float).reshape(-1)
    if x_final.size < 7:
        raise ValueError("Final state must contain at least 7 elements (x,y,z,vx,vy,vz,Cr).")
    r_km = x_final[:3]
    v_km_s = x_final[3:6]
    cr = float(x_final[6])
    print(f"{label} Final State:")
    print(f"  Position [km]: [{r_km[0]:.6f}, {r_km[1]:.6f}, {r_km[2]:.6f}]")
    print(f"  Velocity [km/s]: [{v_km_s[0]:.9f}, {v_km_s[1]:.9f}, {v_km_s[2]:.9f}]")
    print(f"  Cr [-]: {cr:.9f}")


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


def load_project2_obs_for_filter(obs_path: Path) -> list[dict]:
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
    all_meas = load_project2_obs_for_filter(obs_path)
    stations = build_stations()

    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_ekf = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    x0_ekf = np.array(
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
    P0_ekf = np.diag(
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

    # SNC
    sigma_acc_km_s2 = 1.0e-8
    Q_ekf = np.diag([sigma_acc_km_s2**2, sigma_acc_km_s2**2, sigma_acc_km_s2**2])

    dyn_ekf = lambda tau, x: mu_sun_srp_state_deriv(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac_ekf(tau, x):
        r_earth_km, _ = earth_state_func(tau)
        r_sun_km, _ = sun_state_func(tau)
        return srp_thirdbody_variational_eq(
            r_sc=np.asarray(x[:3], dtype=float),
            r_earth=np.asarray(r_earth_km, dtype=float),
            r_sun=np.asarray(r_sun_km, dtype=float),
            Cr=float(x[6]),
            area=float(scConst.area),
            mass=float(scConst.mass),
            mu_earth=float(pConst.mu_earth),
            mu_i=float(pConst.mu_sun),
            solar_flux_1au=float(scConst.solar_flux_1au),
            c=float(scConst.c),
            AU_m=float(scConst.AU_m),
        )

    ekf = ExtendedKalmanFilter(
        x0=x0_ekf,
        P0=P0_ekf,
        R=R_ekf,
        Q=Q_ekf,
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
        dyn_fun=dyn_ekf,
        dyn_jac=jac_ekf,
    )
    if hasattr(ekf, "_delegate"):
        ekf._delegate.q_frame = "eci"
    else:
        ekf.q_frame = "eci"

    out_iekf = ekf.run_iterated(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=None,
        max_iters=10,
        bound_level=5.0,
        iter_tol=1.0e-8,
        show_progress=True,
        progress_every=50,
    )
    print(f"Iterated EKF Run Complete. Number Of Updates: {len(out_iekf['t_meas'])}")

    result_iekf = run_filter_post_processing_18(
        out=out_iekf,
        length_unit_in="km",
        length_unit_out="m",
    )

    outdir_base = Path(__file__).resolve().parent / "Plots" / "Main IEKF"
    outdir_final = outdir_base / "Final IEKF Forward"
    outdir_final.mkdir(parents=True, exist_ok=True)

    make_prefit_residuals_plot(result_iekf, outdir_final)
    make_postfit_residuals_linear_plot(result_iekf, outdir_final)
    make_trace_cov_plot(result_iekf, outdir_final)
    make_trace_cov_pos_vel_plot(result_iekf, outdir_final, length_unit="m")
    make_final_state_estimate_plot(out_iekf, outdir_final)

    # Smooth the provided IEKF forward pass.
    out_smooth = ekf.run_smoother(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=None,
        forward_out=out_iekf,
        show_progress=True,
        progress_every=50,
    )
    print(f"IEKF Smoother Run Complete. Number Of Updates: {len(out_smooth['t_meas'])}")

    # Smoothed-back plots
    out_smooth_for_plots = dict(out_smooth)
    out_smooth_for_plots["Xhat_meas"] = out_smooth["Xhat_smooth"]
    out_smooth_for_plots["xhat_meas"] = out_smooth["Xhat_smooth"]
    out_smooth_for_plots["X_pf"] = out_smooth["Xhat_smooth"]
    out_smooth_for_plots["P_meas"] = out_smooth["P_smooth"]
    out_smooth_for_plots["two_sigma_meas"] = out_smooth.get("two_sigma_smooth", out_smooth["two_sigma_meas"])
    if out_smooth.get("state_error_smooth_meas", None) is not None:
        out_smooth_for_plots["state_error_meas"] = out_smooth["state_error_smooth_meas"]
    n = out_smooth["P_smooth"].shape[1]
    out_smooth_for_plots["P_pf"] = out_smooth["P_smooth"].reshape(len(out_smooth["t_meas"]), n * n)
    out_smooth_for_plots["prefit_resids_final"] = out_smooth["prefit_resids_final"]
    out_smooth_for_plots["postfit_resids_linear_final"] = out_smooth.get(
        "postfit_resids_linear_smooth",
        out_smooth.get("postfit_resids_smooth", out_smooth["postfit_resids_linear_final"]),
    )
    out_smooth_for_plots["postfit_resids_meas"] = out_smooth_for_plots["postfit_resids_linear_final"]

    result_smooth = run_filter_post_processing_18(
        out=out_smooth_for_plots,
        length_unit_in="km",
        length_unit_out="m",
    )

    outdir_smooth = outdir_base / "Smoothed Back"
    outdir_smooth.mkdir(parents=True, exist_ok=True)

    make_prefit_residuals_plot(result_smooth, outdir_smooth)
    make_postfit_residuals_linear_plot(result_smooth, outdir_smooth)
    make_trace_cov_plot(result_smooth, outdir_smooth)
    make_trace_cov_pos_vel_plot(result_smooth, outdir_smooth, length_unit="m")
    make_final_state_estimate_plot(out_smooth_for_plots, outdir_smooth)

    iters = np.asarray(out_iekf.get("iter_counts", []), dtype=float)
    if iters.size > 0:
        print(
            f"IEKF Iteration Stats: Mean={np.nanmean(iters):.3f}, "
            f"Max={int(np.nanmax(iters))}, Min={int(np.nanmin(iters))}"
        )

    print_final_state_summary("IEKF", out_smooth["xhat_meas"][-1])
    print(f"Saved Final IEKF Plots To: {outdir_final}")
    print(f"Saved Smoothed Plots To: {outdir_smooth}")


if __name__ == "__main__":
    main()
