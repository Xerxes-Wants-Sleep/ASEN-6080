import csv
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append("../../")

from src.Functions.filters import LinearizedKalmanFilter
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_state_deriv
from src.Functions.jacobians import srp_thirdbody_variational_eq
from src.Functions.range_rangerate import H_range_rangerate
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


def generic_rts_smoother(
    x_filt_hist: np.ndarray,
    P_filt_hist: np.ndarray,
    x_pred_hist: np.ndarray,
    P_pred_hist: np.ndarray,
    Phi_step_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_filt_hist = np.asarray(x_filt_hist, dtype=float)
    P_filt_hist = np.asarray(P_filt_hist, dtype=float)
    x_pred_hist = np.asarray(x_pred_hist, dtype=float)
    P_pred_hist = np.asarray(P_pred_hist, dtype=float)
    Phi_step_hist = np.asarray(Phi_step_hist, dtype=float)

    N, n = x_filt_hist.shape
    x_smooth = x_filt_hist.copy()
    P_smooth = P_filt_hist.copy()
    if N == 0:
        return x_smooth, P_smooth

    I_n = np.eye(n, dtype=float)
    for k in range(N - 2, -1, -1):
        Phi = Phi_step_hist[k + 1]
        P_k_k = P_filt_hist[k]
        P_kp1_k = P_pred_hist[k + 1]
        invP = np.linalg.solve(P_kp1_k, I_n)
        Ck = (P_k_k @ Phi.T) @ invP
        x_smooth[k] = x_filt_hist[k] + Ck @ (x_smooth[k + 1] - x_pred_hist[k + 1])
        P_smooth[k] = P_filt_hist[k] + Ck @ (P_smooth[k + 1] - P_pred_hist[k + 1]) @ Ck.T
        P_smooth[k] = 0.5 * (P_smooth[k] + P_smooth[k].T)

    return x_smooth, P_smooth


def reconstruct_pred_histories(
    *,
    lkf: LinearizedKalmanFilter,
    out_fwd: dict,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(out_fwd["t_meas"], dtype=float)
    Xstar = np.asarray(out_fwd["Xstar_hist"], dtype=float)
    Phi_step = np.asarray(out_fwd["Phi_step"], dtype=float)
    x_filt = np.asarray(out_fwd["x_filt_hist"], dtype=float)
    P_filt = np.asarray(out_fwd["P_meas"], dtype=float)

    N, n = x_filt.shape
    x_pred = np.zeros((N, n), dtype=float)
    P_pred = np.zeros((N, n, n), dtype=float)

    x_pred[0] = x_filt[0]
    P_pred[0] = P_filt[0]

    # For 7-state Project 2 runs, LinearizedKalmanFilter delegates to
    # LinearizedKalmanFilter18State. Process-noise lives on the delegate.
    q_owner = lkf._delegate if hasattr(lkf, "_delegate") else lkf

    for k in range(1, N):
        dt = float(t[k] - t[k - 1])
        Qk = q_owner.build_process_noise(dt, r_eci=Xstar[k, :3], v_eci=Xstar[k, 3:6])
        x_pred[k] = Phi_step[k] @ x_filt[k - 1]
        P_pred[k] = Phi_step[k] @ P_filt[k - 1] @ Phi_step[k].T + Qk
        P_pred[k] = 0.5 * (P_pred[k] + P_pred[k].T)

    return x_pred, P_pred


def build_smoothed_plot_out(
    *,
    out_fwd: dict,
    x_smooth_err: np.ndarray,
    P_smooth: np.ndarray,
    meas_subset: list[dict],
    stations: list[Stations],
) -> dict:
    Xstar = np.asarray(out_fwd["Xstar_hist"], dtype=float)
    X_smooth = Xstar + np.asarray(x_smooth_err, dtype=float)
    prefit = np.asarray(out_fwd["prefit_resids_final"], dtype=float)

    N = X_smooth.shape[0]
    n = X_smooth.shape[1]
    station_map = {st.name: st for st in stations}
    postfit_lin_smooth = np.full((N, 2), np.nan, dtype=float)

    for i, m in enumerate(meas_subset):
        st = station_map[m["station"]]
        t = float(m["t"])
        if not np.all(np.isfinite(prefit[i, :])):
            continue
        Rs, Vs, _ = st.ecef2eci(t, st.r_ecef, np.zeros(3))
        H6 = H_range_rangerate(Xstar[i, :3], Xstar[i, 3:6], Rs, Vs)
        H = np.zeros((2, n), dtype=float)
        H[:, :6] = H6
        postfit_lin_smooth[i, :] = prefit[i, :] - (H @ np.asarray(x_smooth_err[i, :], dtype=float))

    out_smooth = dict(out_fwd)
    out_smooth["Xhat_meas"] = X_smooth
    out_smooth["xhat_meas"] = X_smooth
    out_smooth["X_pf"] = X_smooth
    out_smooth["P_meas"] = np.asarray(P_smooth, dtype=float)
    out_smooth["P_pf"] = np.asarray(P_smooth, dtype=float).reshape(N, n * n)
    out_smooth["two_sigma_meas"] = 2.0 * np.sqrt(
        np.maximum(np.diagonal(np.asarray(P_smooth, dtype=float), axis1=1, axis2=2), 0.0)
    )
    out_smooth["prefit_resids_final"] = prefit
    out_smooth["postfit_resids_linear_final"] = postfit_lin_smooth
    out_smooth["postfit_resids_meas"] = postfit_lin_smooth

    return out_smooth


def make_standard_plot_set(out: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    result = run_filter_post_processing_18(
        out=out,
        length_unit_in="km",
        length_unit_out="m",
    )
    make_prefit_residuals_plot(result, outdir)
    make_postfit_residuals_linear_plot(result, outdir)
    make_trace_cov_plot(result, outdir)
    make_trace_cov_pos_vel_plot(result, outdir, length_unit="m")
    make_final_state_estimate_plot(out, outdir)


def main():
    obs_path = Path(__file__).resolve().parent / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs_for_filter(obs_path)
    stations = build_stations()

    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_lkf = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    x0_current = np.array(
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
    P0_lkf = np.diag(
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

    sigma_acc_km_s2 = 5.0e-10
    Q_lkf_snc = np.diag(
        [
            sigma_acc_km_s2**2,
            sigma_acc_km_s2**2,
            sigma_acc_km_s2**2,
        ]
    )

    dyn_lkf = lambda tau, x: mu_sun_srp_state_deriv(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac_lkf(tau, x):
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

    max_t = float(np.max([m["t"] for m in all_meas]))
    max_day = max_t / 86400.0

    arc_days = list(np.arange(50.0, max_day + 1.0e-9, 50.0))
    if len(arc_days) == 0 or (not np.isclose(arc_days[-1], max_day)):
        arc_days.append(max_day)

    out_root = Path(__file__).resolve().parent / "Plots" / "Main Broken ILKF"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    final_forward = None
    final_smooth = None
    t_prev_end = -np.inf

    for arc_idx, day_end in enumerate(arc_days, start=1):
        t_end = float(day_end * 86400.0)
        meas_arc = [m for m in all_meas if t_prev_end < float(m["t"]) <= t_end]
        if len(meas_arc) == 0:
            continue

        t_start_days = (0.0 if not np.isfinite(t_prev_end) else (t_prev_end / 86400.0))
        print(
            f"\nRunning Arc {arc_idx}: {t_start_days:.3f} -> {day_end:.3f} days "
            f"({len(meas_arc)} measurements)"
        )

        lkf = LinearizedKalmanFilter(
            X0_star=x0_current,
            P0=P0_lkf,
            R=R_lkf,
            Q=Q_lkf_snc,
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
        if hasattr(lkf, "_delegate"):
            lkf._delegate.q_frame = "eci"
        else:
            lkf.q_frame = "eci"

        out_fwd = lkf.run_iterated(
            all_meas=meas_arc,
            stations=stations,
            Xtrue_meas=None,
            max_iter=10,
            tol=1.0e-10,
            reset_P0_each_iter=True,
            verbose=True,
        )

        x_pred_hist, P_pred_hist = reconstruct_pred_histories(lkf=lkf, out_fwd=out_fwd)
        x_filt_hist = np.asarray(out_fwd["x_filt_hist"], dtype=float)
        P_filt_hist = np.asarray(out_fwd["P_meas"], dtype=float)
        Phi_step = np.asarray(out_fwd["Phi_step"], dtype=float)

        x_smooth_err, P_smooth = generic_rts_smoother(
            x_filt_hist=x_filt_hist,
            P_filt_hist=P_filt_hist,
            x_pred_hist=x_pred_hist,
            P_pred_hist=P_pred_hist,
            Phi_step_hist=Phi_step,
        )
        out_smooth = build_smoothed_plot_out(
            out_fwd=out_fwd,
            x_smooth_err=x_smooth_err,
            P_smooth=P_smooth,
            meas_subset=meas_arc,
            stations=stations,
        )

        # Handoff from the forward posterior at the arc boundary.
        x0_next = np.asarray(out_fwd["Xhat_meas"][-1], dtype=float).copy()
        P0_next = np.asarray(out_fwd["P_meas"][-1], dtype=float).copy()
        if not np.all(np.isfinite(x0_next)):
            x0_fallback = np.asarray(out_smooth["xhat_meas"][0], dtype=float).copy()
            if np.all(np.isfinite(x0_fallback)):
                print(
                    f"Warning: Arc {arc_idx} produced non-finite forward arc-end state; "
                    "falling back to smoothed x0 for next arc."
                )
                x0_next = x0_fallback
            else:
                print(
                    f"Warning: Arc {arc_idx} produced non-finite forward and smoothed states; "
                    "reusing previous x0_current for next arc."
                )
                x0_next = np.asarray(x0_current, dtype=float).copy()
                P0_next = np.asarray(P0_lkf, dtype=float).copy()

        arc_dir = out_root / f"Arc_{arc_idx:02d}_{day_end:07.3f}_Days"
        make_standard_plot_set(out_fwd, arc_dir / "Forward Iterated LKF")
        make_standard_plot_set(out_smooth, arc_dir / "Smoothed Back")

        iter_meta = out_fwd.get("iter_lkf", {})
        summary_rows.append(
            {
                "arc_index": arc_idx,
                "arc_end_days": day_end,
                "num_meas": len(meas_arc),
                "outer_iters": iter_meta.get("num_outer_iters", np.nan),
                "converged": iter_meta.get("converged", False),
                "x0_in_0": float(x0_current[0]),
                "x0_in_1": float(x0_current[1]),
                "x0_in_2": float(x0_current[2]),
                "x0_in_3": float(x0_current[3]),
                "x0_in_4": float(x0_current[4]),
                "x0_in_5": float(x0_current[5]),
                "x0_in_6": float(x0_current[6]),
                "x0_out_0": float(x0_next[0]),
                "x0_out_1": float(x0_next[1]),
                "x0_out_2": float(x0_next[2]),
                "x0_out_3": float(x0_next[3]),
                "x0_out_4": float(x0_next[4]),
                "x0_out_5": float(x0_next[5]),
                "x0_out_6": float(x0_next[6]),
            }
        )

        print(
            f"Arc {arc_idx} Complete: End Day={day_end:.3f}, "
            f"Outer Iter={iter_meta.get('num_outer_iters', 'n/a')}, "
            f"Converged={iter_meta.get('converged', False)}"
        )

        x0_current = x0_next
        P0_lkf = P0_next
        t_prev_end = t_end
        final_forward = out_fwd
        final_smooth = out_smooth

    # Whole-run plots (from final/full arc)
    whole_dir = out_root / "Whole Run"
    if final_forward is not None and final_smooth is not None:
        make_standard_plot_set(final_forward, whole_dir / "Forward Iterated LKF")
        make_standard_plot_set(final_smooth, whole_dir / "Smoothed Back")
        print_final_state_summary("Broken ILKF Whole Run (Smoothed)", final_smooth["xhat_meas"][-1])

    # Save arc-by-arc x0 update summary
    summary_path = out_root / "arc_x0_update_summary.csv"
    if len(summary_rows) > 0:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved Arc Subfolder Plots Under: {out_root}")
    print(f"Saved x0 Update Summary To: {summary_path}")


if __name__ == "__main__":
    main()
