import numpy as np
import pandas as pd
import sys
import json
from pathlib import Path

sys.path.append("../../")

from src.Functions.filters import UnscentedKalmanFilter, ExtendedKalmanFilter, KalmanFilterBase
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_state_deriv
from src.Functions.jacobians import srp_thirdbody_variational_eq
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing_18
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
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


def concat_filter_outputs(parts: list[dict]) -> dict:
    if len(parts) == 0:
        raise ValueError("No filter outputs to concatenate.")

    t_meas = np.concatenate([np.asarray(p["t_meas"], dtype=float) for p in parts], axis=0)
    station_meas = []
    for p in parts:
        station_meas.extend(list(p["station_meas"]))

    xhat_meas = np.concatenate([np.asarray(p["xhat_meas"], dtype=float) for p in parts], axis=0)
    P_meas = np.concatenate([np.asarray(p["P_meas"], dtype=float) for p in parts], axis=0)
    prefit = np.concatenate([np.asarray(p["prefit_resids_final"], dtype=float) for p in parts], axis=0)
    postfit_lin = np.concatenate([np.asarray(p["postfit_resids_linear_final"], dtype=float) for p in parts], axis=0)
    postfit_nl = np.concatenate([np.asarray(p["postfit_resids_meas"], dtype=float) for p in parts], axis=0)

    two_sigma = 2.0 * np.sqrt(np.maximum(np.diagonal(P_meas, axis1=1, axis2=2), 0.0))
    rms_final = KalmanFilterBase.compute_rms_summary_from_arrays(
        t=t_meas,
        postfit=postfit_nl,
        state_err=None,
        first_pass_gap_s=6 * 3600.0,
    )
    R_ref = None
    if len(parts) > 0 and ("R" in parts[0]) and (parts[0]["R"] is not None):
        R_ref = np.asarray(parts[0]["R"], dtype=float)

    return {
        "t_meas": t_meas,
        "station_meas": station_meas,
        "xhat_meas": xhat_meas,
        "Xhat_meas": xhat_meas,
        "P_meas": P_meas,
        "Phat_meas": P_meas,
        "two_sigma_meas": two_sigma,
        "prefit_resids_final": prefit,
        "postfit_resids_linear_final": postfit_lin,
        "postfit_resids_meas": postfit_nl,
        "state_error_meas": None,
        "rms_final": rms_final,
        "rms_by_iter": None,
        "R": R_ref,
    }


def save_ukf_handoff_json(path: Path, t_handoff: float, x_handoff: np.ndarray, P_handoff: np.ndarray, switch_t: float):
    payload = {
        "switch_t_s": float(switch_t),
        "t_handoff_s": float(t_handoff),
        "x_handoff": np.asarray(x_handoff, dtype=float).tolist(),
        "P_handoff": np.asarray(P_handoff, dtype=float).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_ukf_handoff_json(path: Path) -> tuple[float, np.ndarray, np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    t_handoff = float(payload["t_handoff_s"])
    x_handoff = np.asarray(payload["x_handoff"], dtype=float)
    P_handoff = np.asarray(payload["P_handoff"], dtype=float)
    switch_t_saved = float(payload.get("switch_t_s", np.nan))
    return t_handoff, x_handoff, P_handoff, switch_t_saved


def make_section_plots(out_section: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    result = run_filter_post_processing_18(
        out=out_section,
        length_unit_in="km",
        length_unit_out="m",
    )
    make_postfit_residuals_linear_plot(result, outdir)
    make_trace_cov_pos_vel_plot(result, outdir, length_unit="m")
    make_final_state_estimate_plot(out_section, outdir)


def main():
    switch_t = 1.8e7
    # Set True to skip rerunning UKF and reuse saved UKF->EKF handoff.
    reuse_saved_ukf_handoff = False

    obs_path = Path(__file__).resolve().parent / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs(obs_path)
    stations = build_stations()
    outdir_base = Path(__file__).resolve().parent / "Plots" / "Main UKF-EKF"
    handoff_json_path = outdir_base / "ukf_handoff_state.json"

    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_filter = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    x0_common = np.array(
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
    P0_common = np.diag(
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

    dyn_fun = lambda tau, x: mu_sun_srp_state_deriv(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac_fun(tau, x):
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

    # UKF first segment includes measurements at t <= 1.8e7 s.
    meas_ukf = [m for m in all_meas if float(m["t"]) <= switch_t]
    meas_ekf = [m for m in all_meas if float(m["t"]) > switch_t]

    if len(meas_ukf) == 0:
        raise RuntimeError("No measurements in UKF segment (t <= 1.8e7 s).")
    if len(meas_ekf) == 0:
        raise RuntimeError("No measurements in EKF segment (t > 1.8e7 s).")

    print(f"Segment Sizes: UKF={len(meas_ukf)}, EKF={len(meas_ekf)}")

    out_ukf = None
    if reuse_saved_ukf_handoff and handoff_json_path.exists():
        t_handoff, x_handoff, P_handoff, switch_t_saved = load_ukf_handoff_json(handoff_json_path)
        print(f"Loaded UKF Handoff From JSON: {handoff_json_path}")
        if np.isfinite(switch_t_saved) and (not np.isclose(switch_t_saved, switch_t)):
            print(
                f"Warning: JSON switch_t={switch_t_saved:.3f} differs from current switch_t={switch_t:.3f}"
            )
    else:
        # Segment 1: UKF
        # Q_ukf = np.zeros((7, 7), dtype=float)
        sigma_acc_km_s2 = 5.0e-11
        Q_ukf = np.diag([sigma_acc_km_s2**2, sigma_acc_km_s2**2, sigma_acc_km_s2**2])

        ukf = UnscentedKalmanFilter(
            X0=x0_common,
            P0=P0_common,
            R=R_filter,
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
            dyn_fun=dyn_fun,
        )
        ukf.q_frame = "eci"

        out_ukf = ukf.run(
            all_meas=meas_ukf,
            stations=stations,
            Xtrue_meas=None,
            show_progress=True,
            progress_every=50,
        )
        print(f"UKF Segment Complete. Updates: {len(out_ukf['t_meas'])}")

        # Hand off UKF posterior to EKF segment.
        x_handoff = np.asarray(out_ukf["xhat_meas"][-1], dtype=float)
        P_handoff = np.asarray(out_ukf["P_meas"][-1], dtype=float)
        t_handoff = float(out_ukf["t_meas"][-1])

        save_ukf_handoff_json(handoff_json_path, t_handoff, x_handoff, P_handoff, switch_t)
        print(f"Saved UKF Handoff JSON: {handoff_json_path}")

    # Segment 2: EKF with SNC
    sigma_acc_ekf_km_s2 = 1.0e-8
    Q_ekf_snc = np.diag(
        [
            sigma_acc_ekf_km_s2**2,
            sigma_acc_ekf_km_s2**2,
            sigma_acc_ekf_km_s2**2,
        ]
    )

    ekf = ExtendedKalmanFilter(
        x0=x_handoff,
        P0=P_handoff,
        R=R_filter,
        Q=Q_ekf_snc,
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
        dyn_fun=dyn_fun,
        dyn_jac=jac_fun,
    )
    if hasattr(ekf, "_delegate"):
        ekf._delegate.q_frame = "eci"
    else:
        ekf.q_frame = "eci"

    out_ekf = ekf.run(
        all_meas=meas_ekf,
        stations=stations,
        Xtrue_meas=None,
        t_prev_init=t_handoff,
        show_progress=True,
        progress_every=50,
    )
    print(f"EKF Segment Complete. Updates: {len(out_ekf['t_meas'])}")

    outdir_ukf = outdir_base / "UKF Section"
    outdir_ekf = outdir_base / "EKF Section"
    outdir_total = outdir_base / "Total"

    # Always create EKF section plots.
    make_section_plots(out_ekf, outdir_ekf)
    print(f"Saved EKF Section Plots To: {outdir_ekf}")

    if out_ukf is not None:
        make_section_plots(out_ukf, outdir_ukf)
        print(f"Saved UKF Section Plots To: {outdir_ukf}")

        out_comb = concat_filter_outputs([out_ukf, out_ekf])
        print(f"Combined Run Complete. Total Updates: {len(out_comb['t_meas'])}")
        make_section_plots(out_comb, outdir_total)
        print(f"Saved Total Plots To: {outdir_total}")
        print_final_state_summary("UKF-EKF Total", out_comb["xhat_meas"][-1])
    else:
        print(
            "Skipped UKF run using cached handoff JSON; UKF/Total section plots "
            "were not generated in this run."
        )
        print_final_state_summary("EKF Section", out_ekf["xhat_meas"][-1])


if __name__ == "__main__":
    main()
