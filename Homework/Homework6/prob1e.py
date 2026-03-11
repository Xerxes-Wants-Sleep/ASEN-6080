import contextlib
import io
import sys
from pathlib import Path

sys.path.append("../../")

from prob1b import build_q_accel_cov_km, convert_filter_output_km_to_m_for_plotting, load_hw3_inputs
from src.Functions.filters import UnscentedKalmanFilter
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_postfit_residuals_nonlinear import make_postfit_residuals_nonlinear_plot
from src.helpers.plotting.plot_state_errors_eci import make_state_errors_eci_plots
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0 = load_hw3_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    # Same optimal SNC baseline used previously (from HW3 EKF sweep).
    sigma_q_m_s2 = 3.8311868495572855e-12
    Q_accel = build_q_accel_cov_km(sigma_q_m_s2)

    # Same tuning as the stable UKF configuration from prob1b, but now with J3 ON.
    ukf = UnscentedKalmanFilter(
        X0=x0,
        P0=P0,
        R=R,
        Q=Q_accel,
        mu=mu,
        J2=J2,
        J3=J3,
        Re=6378.0,
        alpha=1.0,
        beta=2.0,
        kappa=None,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
        j2=True,
        j3=True,
        first_pass_gap_s=6 * 3600.0,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        out = ukf.run(all_meas=all_meas, stations=stations, Xtrue_meas=Xtrue_meas)

    result = convert_filter_output_km_to_m_for_plotting(out, R_km=R)

    plot_dir = Path(__file__).resolve().parent / "Plots" / "prob1e-Take2"
    plot_dir.mkdir(parents=True, exist_ok=True)

    make_state_errors_eci_plots(result, plot_dir, zoom_t0_s=30000.0)
    make_postfit_residuals_linear_plot(result, plot_dir)
    make_postfit_residuals_nonlinear_plot(result, plot_dir)
    make_trace_cov_pos_vel_plot(
        result,
        plot_dir,
        title=f"UKF prob1e (J3 ON): trace(P_pos), trace(P_vel), sigma_q={sigma_q_m_s2:.3e} m/s^2",
    )

    print(f"Saved UKF prob1e plots to: {plot_dir}")


if __name__ == "__main__":
    main()
