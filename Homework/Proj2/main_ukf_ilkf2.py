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
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_prefit_residuals import make_prefit_residuals_plot
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


def make_segment_plots(out: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    result = run_filter_post_processing_18(
        out=out,
        length_unit_in="km",
        length_unit_out="m",
    )
    make_postfit_residuals_linear_plot(result, outdir)
    make_final_state_estimate_plot(out, outdir)


def run_ukf_segment(
    meas: list[dict],
    stations,
    x0: np.ndarray,
    P0: np.ndarray,
    R: np.ndarray,
    dyn_fun,
    *,
    Q: np.ndarray,
    show_progress: bool = True,
) -> dict:
    ukf = UnscentedKalmanFilter(
        X0=x0,
        P0=P0,
        R=R,
        Q=Q,
        mu=398600.4415,
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
    return ukf.run(
        all_meas=meas,
        stations=stations,
        Xtrue_meas=None,
        show_progress=show_progress,
        progress_every=50,
    )


def run_iter_lkf_segment(
    meas: list[dict],
    stations,
    x0_star: np.ndarray,
    P0: np.ndarray,
    R: np.ndarray,
    dyn_fun,
    jac_fun,
    *,
    Q_lkf: np.ndarray,
    max_iter: int = 6,
    tol: float = 1.0e-8,
) -> dict:
    lkf = LinearizedKalmanFilter(
        X0_star=x0_star,
        P0=P0,
        R=R,
        Q=Q_lkf,
        mu=398600.4415,
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

    return lkf.run_iterated(
        all_meas=meas,
        stations=stations,
        Xtrue_meas=None,
        max_iter=max_iter,
        tol=tol,
        reset_P0_each_iter=True,
        verbose=True,
    )


def main():
    t1_end = 18863968.0
    t2_start = 19123768.0
    t2_end = 19125628.0
    t3_end = 19135168.0
    t4_start = 19379968.0

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

    # Measurement segments (4 pieces for the effective run):
    #  1) UKF  : t <= 18863968
    #  2) ILKF : 19123768 <= t <= 19125628
    #  3) UKF  : 19125628 < t <= 19135168
    #  4) UKF  : t >= 19379968, initialized from ILKF pre-pass on same chunk
    meas1 = [m for m in all_meas if float(m["t"]) <= t1_end]
    meas2 = [m for m in all_meas if t2_start <= float(m["t"]) <= t2_end]
    meas3 = [m for m in all_meas if t2_end < float(m["t"]) <= t3_end]
    meas4 = [m for m in all_meas if float(m["t"]) >= t4_start]

    if len(meas1) == 0 or len(meas2) == 0 or len(meas3) == 0 or len(meas4) == 0:
        raise RuntimeError("One or more requested measurement segments are empty. Check time bounds.")

    print(
        f"Segment sizes: S1={len(meas1)} | S2={len(meas2)} | S3={len(meas3)} | S4={len(meas4)}"
    )

    out_root = Path(__file__).resolve().parent / "Plots" / "Main UKF-ILKF2"

    # Segment 1: UKF with zero SNC
    Q_ukf_zero = np.zeros((7, 7), dtype=float)
    out1 = run_ukf_segment(
        meas1,
        stations,
        x0_common,
        P0_common,
        R_filter,
        dyn_fun,
        Q=Q_ukf_zero,
        show_progress=True,
    )
    print(f"Segment 1 (UKF) complete: {len(out1['t_meas'])} updates")
    make_segment_plots(out1, out_root / "Segment 1 UKF")

    # Handoff propagate to Segment 2 start
    x1f = np.asarray(out1["xhat_meas"][-1], dtype=float)
    P1f = np.asarray(out1["P_meas"][-1], dtype=float)
    t1f = float(out1["t_meas"][-1])
    x2_0, P2_0 = propagate_state_cov(x1f, P1f, t1f, float(meas2[0]["t"]), dyn_fun, jac_fun)

    # Segment 2: ILKF with large SNC
    sigma_acc_lkf_km_s2 = 1.0e-8
    Q_lkf_large = np.diag(
        [
            sigma_acc_lkf_km_s2**2,
            sigma_acc_lkf_km_s2**2,
            sigma_acc_lkf_km_s2**2,
        ]
    )
    out2 = run_iter_lkf_segment(
        meas2,
        stations,
        x2_0,
        P2_0,
        R_filter,
        dyn_fun,
        jac_fun,
        Q_lkf=Q_lkf_large,
        max_iter=6,
        tol=1.0e-8,
    )
    print(f"Segment 2 (ILKF) complete: {len(out2['t_meas'])} updates")
    make_segment_plots(out2, out_root / "Segment 2 ILKF")

    # Handoff propagate to Segment 3 start
    x2f = np.asarray(out2["xhat_meas"][-1], dtype=float)
    P2f = np.asarray(out2["P_meas"][-1], dtype=float)
    t2f = float(out2["t_meas"][-1])
    x3_0, P3_0 = propagate_state_cov(x2f, P2f, t2f, float(meas3[0]["t"]), dyn_fun, jac_fun)

    # Segment 3: UKF with zero SNC
    out3 = run_ukf_segment(
        meas3,
        stations,
        x3_0,
        P3_0,
        R_filter,
        dyn_fun,
        Q=Q_ukf_zero,
        show_progress=True,
    )
    print(f"Segment 3 (UKF) complete: {len(out3['t_meas'])} updates")
    make_segment_plots(out3, out_root / "Segment 3 UKF")

    # Handoff propagate to Segment 4 start
    x3f = np.asarray(out3["xhat_meas"][-1], dtype=float)
    P3f = np.asarray(out3["P_meas"][-1], dtype=float)
    t3f = float(out3["t_meas"][-1])
    x4_0, P4_0 = propagate_state_cov(x3f, P3f, t3f, float(meas4[0]["t"]), dyn_fun, jac_fun)

    # Segment 4 pre-pass: ILKF on final chunk to refine start/state stats
    out4_ilkf = run_iter_lkf_segment(
        meas4,
        stations,
        x4_0,
        P4_0,
        R_filter,
        dyn_fun,
        jac_fun,
        Q_lkf=Q_lkf_large,
        max_iter=6,
        tol=1.0e-8,
    )
    print(f"Segment 4 pre-pass (ILKF) complete: {len(out4_ilkf['t_meas'])} updates")
    make_segment_plots(out4_ilkf, out_root / "Segment 4 ILKF Prepass")

    # Final Segment 4 effective run: hand off at 19379968 back to UKF and run full final chunk
    x4_start_from_ilkf = np.asarray(out4_ilkf["xhat_meas"][0], dtype=float)
    P4_start_from_ilkf = np.asarray(out4_ilkf["P_meas"][0], dtype=float)
    out4 = run_ukf_segment(
        meas4,
        stations,
        x4_start_from_ilkf,
        P4_start_from_ilkf,
        R_filter,
        dyn_fun,
        Q=Q_ukf_zero,
        show_progress=True,
    )
    print(f"Segment 4 final (UKF) complete: {len(out4['t_meas'])} updates")
    make_segment_plots(out4, out_root / "Segment 4 UKF Final")

    # Total combined output from effective 4 pieces
    out_total = concat_filter_outputs([out1, out2, out3, out4], R_filter)
    print(f"Total combined updates: {len(out_total['t_meas'])}")

    result_total = run_filter_post_processing_18(
        out=out_total,
        length_unit_in="km",
        length_unit_out="m",
    )

    out_total_dir = out_root / "Total Run"
    out_total_dir.mkdir(parents=True, exist_ok=True)
    make_prefit_residuals_plot(result_total, out_total_dir)
    make_postfit_residuals_linear_plot(result_total, out_total_dir)
    make_trace_cov_plot(result_total, out_total_dir)
    make_trace_cov_pos_vel_plot(result_total, out_total_dir, length_unit="m")
    make_final_state_estimate_plot(out_total, out_total_dir)

    print_final_state_summary("UKF-ILKF2 Total", out_total["xhat_meas"][-1])
    print(f"Saved total-run plots to: {out_total_dir}")
    print(f"Saved segment subfolder plots under: {out_root}")


if __name__ == "__main__":
    main()
