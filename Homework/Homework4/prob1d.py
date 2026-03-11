import sys
from pathlib import Path

import numpy as np

sys.path.append("../../")

from prob1c import load_problem2_inputs
from src.Functions.filters import LinearizedKalmanFilter
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_state_errors_eci import make_state_errors_eci_plots
from src.helpers.plotting.post_processing import print_rms_summary, run_filter_post_processing


def build_q_accel_cov_km(sigma_m_s2: float) -> np.ndarray:
    sigma_km_s2 = float(sigma_m_s2) * 1.0e-3
    return np.diag([sigma_km_s2**2, sigma_km_s2**2, sigma_km_s2**2])


def run_lkf_snc_smoothed(
    *,
    all_meas,
    stations,
    Xtrue_meas,
    x0_star,
    P0,
    R,
    mu,
    J2,
    J3,
    sigma_opt_m_s2: float,
):
    lkf = LinearizedKalmanFilter(
        X0_star=x0_star,
        P0=P0,
        R=R,
        Q=build_q_accel_cov_km(sigma_opt_m_s2),
        mu=mu,
        J2=J2,
        J3=J3,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
        j2=True,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
    )

    out = lkf.run(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        run_smoother=True,
        smooth_back_points=None,
    )

    if not bool(out.get("smoother_ran", False)):
        raise RuntimeError("LKF smoother did not run in prob1d.")
    return out


def convert_smoothed_lkf_output_to_plot_result(out: dict, *, R_km: np.ndarray):
    d = dict(out)

    d["xhat_meas"] = np.asarray(d["Xhat_smooth"], dtype=float)
    d["Xhat_meas"] = np.asarray(d["Xhat_smooth"], dtype=float)
    d["P_meas"] = np.asarray(d["P_smooth"], dtype=float)
    d["Phat_meas"] = np.asarray(d["P_smooth"], dtype=float)
    d["two_sigma_meas"] = np.asarray(d["two_sigma_smooth"], dtype=float)
    d["state_error_meas"] = np.asarray(d["state_error_smooth_meas"], dtype=float)
    d["postfit_resids_linear_final"] = np.asarray(d["postfit_resids_linear_smooth"], dtype=float)

    # Convert known LKF units (km, km/s) to plotting units (m, m/s).
    d["xhat_meas"] = d["xhat_meas"] * 1000.0
    d["Xhat_meas"] = d["Xhat_meas"] * 1000.0
    d["X_pf"] = np.asarray(d["X_pf"], dtype=float) * 1000.0
    d["state_error_meas"] = d["state_error_meas"] * 1000.0

    d["P_meas"] = d["P_meas"] * (1000.0**2)
    d["Phat_meas"] = d["Phat_meas"] * (1000.0**2)
    d["P_pf"] = np.asarray(d["P_pf"], dtype=float) * (1000.0**2)

    # Note: key name is legacy; in your smoother this is actually 3-sigma.
    d["two_sigma_meas"] = d["two_sigma_meas"] * 1000.0

    d["prefit_resids_final"] = np.asarray(d["prefit_resids_final"], dtype=float) * 1000.0
    d["postfit_resids_linear_final"] = np.asarray(d["postfit_resids_linear_final"], dtype=float) * 1000.0
    d["postfit_resids_meas"] = np.asarray(d["postfit_resids_meas"], dtype=float) * 1000.0
    d["R"] = np.asarray(R_km, dtype=float) * (1000.0**2)

    return run_filter_post_processing(out=d)


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0_star = load_problem2_inputs()
    sigma_opt_m_s2 = 1.668e-4

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    out = run_lkf_snc_smoothed(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0_star=x0_star,
        P0=P0,
        R=R,
        mu=mu,
        J2=J2,
        J3=J3,
        sigma_opt_m_s2=sigma_opt_m_s2,
    )

    result = convert_smoothed_lkf_output_to_plot_result(out, R_km=R)

    plot_dir = Path(__file__).resolve().parent / "Plots" / "Prob1d"
    plot_dir.mkdir(parents=True, exist_ok=True)

    make_state_errors_eci_plots(result, plot_dir)
    make_postfit_residuals_linear_plot(result, plot_dir)

    print("\nProb 1d: Smoothed CKF/LKF with SNC (using HW3 optimal sigma)")
    print(f"  sigma_opt = {sigma_opt_m_s2:.6e} m/s^2")
    print_rms_summary(result, ignore_first_pass=False)
    print(f"\nSaved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
