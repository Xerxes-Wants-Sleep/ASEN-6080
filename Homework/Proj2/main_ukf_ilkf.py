import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append("../../")

from src.Functions.filters import UnscentedKalmanFilter, LinearizedKalmanFilter, KalmanFilterBase
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_state_deriv
from src.Functions.jacobians import srp_thirdbody_variational_eq
from src.Functions.Ephem import ephem
from src.Functions.propagation import propagate_x_phi_step
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


def propagate_state_cov(
    x0: np.ndarray,
    P0: np.ndarray,
    t0: float,
    t1: float,
    dyn_fun,
    jac_fun,
):
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    P0 = np.asarray(P0, dtype=float)
    t0 = float(t0)
    t1 = float(t1)

    if np.isclose(t0, t1):
        return x0.copy(), P0.copy()

    x1, Phi_10 = propagate_x_phi_step(
        x0=x0,
        t0=t0,
        t1=t1,
        f=dyn_fun,
        A=jac_fun,
    )
    P1 = Phi_10 @ P0 @ Phi_10.T
    P1 = 0.5 * (P1 + P1.T)
    return x1, P1


def concat_filter_outputs(parts: list[dict], R: np.ndarray) -> dict:
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
        "R": np.asarray(R, dtype=float),
    }


def main():
    maneuver_t = 18603600.0
    ilkf_end_t = 2.0e7

    obs_path = Path(__file__).resolve().parent / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs_for_filter(obs_path)
    stations = build_stations()

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

    # Partition measurements:
    #  1) UKF (0-SNC): t < maneuver_t
    #  2) ILKF (large SNC): maneuver_t <= t <= ilkf_end_t
    #  3) UKF (0-SNC): t > ilkf_end_t
    meas_seg1 = [m for m in all_meas if float(m["t"]) < maneuver_t]
    meas_seg2 = [m for m in all_meas if maneuver_t <= float(m["t"]) <= ilkf_end_t]
    meas_seg3 = [m for m in all_meas if float(m["t"]) > ilkf_end_t]

    if len(meas_seg2) == 0:
        raise RuntimeError("No measurements in the ILKF maneuver window segment.")
    if len(meas_seg1) == 0:
        raise RuntimeError("No measurements before maneuver_t for phase-1 UKF.")

    print(
        f"Segment Sizes: UKF1={len(meas_seg1)}, ILKF={len(meas_seg2)}, UKF2={len(meas_seg3)}"
    )

    # --------------------------
    # Segment 1: UKF (Q=0)
    # --------------------------
    Q_ukf_zero = np.zeros((7, 7), dtype=float)
    ukf1 = UnscentedKalmanFilter(
        X0=x0_common,
        P0=P0_common,
        R=R_filter,
        Q=Q_ukf_zero,
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
    ukf1.q_frame = "eci"
    out_ukf1 = ukf1.run(
        all_meas=meas_seg1,
        stations=stations,
        Xtrue_meas=None,
        show_progress=True,
        progress_every=50,
    )
    print(f"UKF Phase 1 Complete. Updates: {len(out_ukf1['t_meas'])}")

    x_handoff_1 = np.asarray(out_ukf1["xhat_meas"][-1], dtype=float)
    P_handoff_1 = np.asarray(out_ukf1["P_meas"][-1], dtype=float)
    t_handoff_1 = float(out_ukf1["t_meas"][-1])
    t_seg2_start = float(meas_seg2[0]["t"])
    x0_lkf, P0_lkf = propagate_state_cov(
        x_handoff_1, P_handoff_1, t_handoff_1, t_seg2_start, dyn_fun, jac_fun
    )

    # --------------------------
    # Segment 2: Iterated LKF (large SNC)
    # --------------------------
    sigma_acc_lkf_km_s2 = 1.0e-8
    Q_lkf_large = np.diag(
        [
            sigma_acc_lkf_km_s2**2,
            sigma_acc_lkf_km_s2**2,
            sigma_acc_lkf_km_s2**2,
        ]
    )

    lkf = LinearizedKalmanFilter(
        X0_star=x0_lkf,
        P0=P0_lkf,
        R=R_filter,
        Q=Q_lkf_large,
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
    if hasattr(lkf, "_delegate"):
        lkf._delegate.q_frame = "eci"
    else:
        lkf.q_frame = "eci"

    out_lkf = lkf.run_iterated(
        all_meas=meas_seg2,
        stations=stations,
        Xtrue_meas=None,
        max_iter=6,
        tol=1.0e-8,
        reset_P0_each_iter=True,
        verbose=True,
    )
    iter_meta = out_lkf.get("iter_lkf", {})
    print(
        f"ILKF Phase Complete. Outer Iterations: {iter_meta.get('num_outer_iters', 'n/a')}, "
        f"Converged: {iter_meta.get('converged', False)}"
    )

    # --------------------------
    # Segment 3: UKF (Q=0) from ILKF posterior
    # --------------------------
    parts = [out_ukf1, out_lkf]
    if len(meas_seg3) > 0:
        x_handoff_2 = np.asarray(out_lkf["xhat_meas"][-1], dtype=float)
        P_handoff_2 = np.asarray(out_lkf["P_meas"][-1], dtype=float)
        t_handoff_2 = float(out_lkf["t_meas"][-1])
        t_seg3_start = float(meas_seg3[0]["t"])
        x0_ukf2, P0_ukf2 = propagate_state_cov(
            x_handoff_2, P_handoff_2, t_handoff_2, t_seg3_start, dyn_fun, jac_fun
        )

        # Final UKF segment uses a modest SNC level.
        sigma_acc_ukf2_km_s2 = 1.0e-11
        Q_ukf2_snc = np.diag(
            [
                sigma_acc_ukf2_km_s2**2,
                sigma_acc_ukf2_km_s2**2,
                sigma_acc_ukf2_km_s2**2,
            ]
        )

        ukf2 = UnscentedKalmanFilter(
            X0=x0_ukf2,
            P0=P0_ukf2,
            R=R_filter,
            Q=Q_ukf2_snc,
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
        ukf2.q_frame = "eci"
        out_ukf2 = ukf2.run(
            all_meas=meas_seg3,
            stations=stations,
            Xtrue_meas=None,
            show_progress=True,
            progress_every=50,
        )
        parts.append(out_ukf2)
        print(f"UKF Phase 3 Complete. Updates: {len(out_ukf2['t_meas'])}")

        # Extra plots for only the final UKF segment.
        outdir_seg3 = Path(__file__).resolve().parent / "Plots" / "Main UKF-ILKF-UKF" / "Final UKF Segment"
        outdir_seg3.mkdir(parents=True, exist_ok=True)
        result_seg3 = run_filter_post_processing_18(
            out=out_ukf2,
            length_unit_in="km",
            length_unit_out="m",
        )
        make_postfit_residuals_linear_plot(result_seg3, outdir_seg3)
        make_final_state_estimate_plot(out_ukf2, outdir_seg3)
        print(f"Saved Final UKF Segment Plots To: {outdir_seg3}")
    else:
        print("No measurements after ILKF end window; skipping UKF phase 3.")

    out_comb = concat_filter_outputs(parts, R=R_filter)
    print(f"Combined Run Complete. Total Updates: {len(out_comb['t_meas'])}")

    result = run_filter_post_processing_18(
        out=out_comb,
        length_unit_in="km",
        length_unit_out="m",
    )

    outdir = Path(__file__).resolve().parent / "Plots" / "Main UKF-ILKF-UKF"
    outdir.mkdir(parents=True, exist_ok=True)

    make_prefit_residuals_plot(result, outdir)
    make_postfit_residuals_linear_plot(result, outdir)
    make_trace_cov_plot(result, outdir)
    make_trace_cov_pos_vel_plot(result, outdir, length_unit="m")
    make_final_state_estimate_plot(out_comb, outdir)

    print_final_state_summary("UKF-ILKF-UKF Combined", out_comb["xhat_meas"][-1])
    print(f"Saved Plots To: {outdir}")


if __name__ == "__main__":
    main()
