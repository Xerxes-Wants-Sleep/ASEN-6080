import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("../../")

from src.Functions.filters import UnscentedKalmanFilter
from src.Functions.stations import Stations
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_postfit_residuals_nonlinear import make_postfit_residuals_nonlinear_plot
from src.helpers.plotting.plot_state_errors_eci import make_state_errors_eci_plots
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.post_processing import run_filter_post_processing


def build_stations():
    theta0_deg = 122.0
    w_earth_rad_per_s = 2.0 * np.pi / (3600.0 * 24.0)
    radius_earth_km = 6378.0

    return [
        Stations(
            "Station 1",
            lat_deg=-35.398333,
            lon_deg=148.981944,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "Station 2",
            lat_deg=40.427222,
            lon_deg=355.749444,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "Station 3",
            lat_deg=35.247164,
            lon_deg=243.205000,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
    ]


def load_hw3_inputs():
    script_dir = Path(__file__).resolve().parent

    meas_path = script_dir.parent / "Homework2" / "meas_data" / "prob2_hw2_measurements_noisy.csv"
    truth_path = script_dir.parent / "Homework1" / "HW2_j3_on_truth.csv"

    df_meas = pd.read_csv(meas_path)
    all_meas = sorted(df_meas.to_dict(orient="records"), key=lambda m: float(m["t"]))
    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)

    truth_df = pd.read_csv(truth_path)
    truth_times = truth_df["t_s"].to_numpy(float)
    truth_states = truth_df[["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
    x0_true = truth_states[0, :]
    Xtrue_meas = np.column_stack([np.interp(t_meas, truth_times, truth_states[:, k]) for k in range(6)])

    stations = build_stations()

    sigma_rho_km = 1.0e-3
    sigma_rhod_km_s = 1.0e-6
    R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    sigma_r0_km = 1.0
    sigma_v0_km_s = 1.0e-3
    P0 = np.diag(
        [
            sigma_r0_km**2,
            sigma_r0_km**2,
            sigma_r0_km**2,
            sigma_v0_km_s**2,
            sigma_v0_km_s**2,
            sigma_v0_km_s**2,
        ]
    )

    # Match HW3 EKF-style initialization.
    dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
    x0 = x0_true + dx

    return all_meas, Xtrue_meas, stations, R, P0, x0


def build_q_accel_cov_km(sigma_m_s2: float) -> np.ndarray:
    sigma_km_s2 = float(sigma_m_s2) * 1.0e-3
    return np.diag([sigma_km_s2**2, sigma_km_s2**2, sigma_km_s2**2])


def convert_filter_output_km_to_m_for_plotting(out: dict, *, R_km: np.ndarray):
    d = dict(out)

    if "xhat_meas" not in d and "Xhat_meas" in d:
        d["xhat_meas"] = d["Xhat_meas"]
    if "P_meas" not in d and "Phat_meas" in d:
        d["P_meas"] = d["Phat_meas"]

    d["xhat_meas"] = np.asarray(d["xhat_meas"], dtype=float) * 1000.0
    d["Xhat_meas"] = np.asarray(d["Xhat_meas"], dtype=float) * 1000.0

    if d.get("X_pf", None) is not None:
        d["X_pf"] = np.asarray(d["X_pf"], dtype=float) * 1000.0
    if d.get("state_error_meas", None) is not None:
        d["state_error_meas"] = np.asarray(d["state_error_meas"], dtype=float) * 1000.0

    d["P_meas"] = np.asarray(d["P_meas"], dtype=float) * (1000.0**2)
    if d.get("Phat_meas", None) is not None:
        d["Phat_meas"] = np.asarray(d["Phat_meas"], dtype=float) * (1000.0**2)
    if d.get("P_pf", None) is not None:
        d["P_pf"] = np.asarray(d["P_pf"], dtype=float) * (1000.0**2)

    if d.get("two_sigma_meas", None) is not None:
        d["two_sigma_meas"] = np.asarray(d["two_sigma_meas"], dtype=float) * 1000.0

    for key in ("prefit_resids_final", "postfit_resids_linear_final", "postfit_resids_meas"):
        if d.get(key, None) is not None:
            d[key] = np.asarray(d[key], dtype=float) * 1000.0

    d["R"] = np.asarray(R_km, dtype=float) * (1000.0**2)

    return run_filter_post_processing(out=d)


def run_ukf_case(
    *,
    all_meas,
    stations,
    Xtrue_meas,
    x0,
    P0,
    R,
    mu,
    J2,
    J3,
    alpha: float,
    beta: float,
    Q: np.ndarray,
):
    ukf = UnscentedKalmanFilter(
        X0=x0,
        P0=P0,
        R=R,
        Q=Q,
        mu=mu,
        J2=J2,
        J3=J3,
        Re=6378.0,
        alpha=alpha,
        beta=beta,
        kappa=None,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
        j2=True,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        return ukf.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0 = load_hw3_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    # Process-noise acceleration std from prior HW3 SNC tuning (m/s^2).
    sigma_q_m_s2 = 3.8311868495572855e-08
    Q_accel = build_q_accel_cov_km(3e-7)
    Q_accel = build_q_accel_cov_km(sigma_q_m_s2)

    Q_accel2 = build_q_accel_cov_km(0)

    cases = [
        # {
        #     "case_id": "i",
        #     "label": "i: alpha=1.0, beta=2.0, Q=0",
        #     "alpha": 1.0,
        #     "beta": 2.0,
        #     "Q": np.zeros((6, 6), dtype=float),
        # },
        # {
        #     "case_id": "ii",
        #     "label": f"ii: alpha=1.0, beta=2.0, Q_accel (sigma={sigma_q_m_s2:.3e} m/s^2)",
        #     "alpha": 1.0,
        #     "beta": 2.0,
        #     "Q": Q_accel,
        # },
        {
            "case_id": "iii",
            "label": f"iii: alpha=1e-4, beta=2.0, Q_accel (sigma={3e-7:.3e} m/s^2)",
            "alpha": 1.0e-4,
            "beta": 2.0,
            "Q": Q_accel2,
        },
    ]

    plot_root = Path(__file__).resolve().parent / "Plots" / "prob1b-5_Q=3e-7"
    plot_root.mkdir(parents=True, exist_ok=True)

    for cfg in cases:
        out = run_ukf_case(
            all_meas=all_meas,
            stations=stations,
            Xtrue_meas=Xtrue_meas,
            x0=x0,
            P0=P0,
            R=R,
            mu=mu,
            J2=J2,
            J3=J3,
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            Q=cfg["Q"],
        )

        result = convert_filter_output_km_to_m_for_plotting(out, R_km=R)
        case_dir = plot_root / cfg["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        make_state_errors_eci_plots(result, case_dir, zoom_t0_s=30000.0)
        make_postfit_residuals_linear_plot(result, case_dir)
        make_postfit_residuals_nonlinear_plot(result, case_dir)
        make_trace_cov_pos_vel_plot(
            result,
            case_dir,
            title=f"UKF {cfg['label']}: trace(P_pos), trace(P_vel)",
        )

        print(f"[{cfg['case_id']}] saved plots to: {case_dir}")

    print(f"\nAll UKF case plots saved under: {plot_root}")


if __name__ == "__main__":
    main()
