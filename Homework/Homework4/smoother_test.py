import sys
from pathlib import Path

import numpy as np

sys.path.append("../../")

from prob1c import load_problem2_inputs
from src.Functions.filters import LinearizedKalmanFilter
from src.helpers.plotting.post_processing import run_filter_post_processing, print_rms_summary
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_state_errors_eci import make_state_errors_eci_plots


def run_lkf_case(
    *,
    all_meas,
    stations,
    Xtrue_meas,
    x0_bar,
    P0,
    R,
    mu,
    J2,
    J3,
    run_smoother: bool,
    smooth_back_points: int | None = None,
):
    Q_zero = np.zeros((6, 6), dtype=float)

    lkf = LinearizedKalmanFilter(
        X0_star=x0_bar,
        P0=P0,
        R=R,
        Q=Q_zero,
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
        run_smoother=run_smoother,
        smooth_back_points=smooth_back_points,
    )

    return out


def to_plotting_result(out: dict, *, use_smoothed_state: bool, R_km: np.ndarray):
    d = dict(out)

    if use_smoothed_state:
        if not bool(d.get("smoother_ran", False)):
            raise RuntimeError("Requested smoothed plotting output, but smoother_ran is False.")
        if d.get("Xhat_smooth", None) is None or d.get("P_smooth", None) is None:
            raise RuntimeError("Smoothed LKF outputs are missing Xhat_smooth and/or P_smooth.")

        # Reuse the existing filter plotting pipeline by swapping in smoothed state/cov histories.
        d["Xhat_meas"] = np.asarray(d["Xhat_smooth"], dtype=float)
        d["xhat_meas"] = np.asarray(d["Xhat_smooth"], dtype=float)
        d["P_meas"] = np.asarray(d["P_smooth"], dtype=float)
        d["Phat_meas"] = np.asarray(d["P_smooth"], dtype=float)
        d["two_sigma_meas"] = d.get("two_sigma_smooth", None)
        d["state_error_meas"] = d.get("state_error_smooth_meas", None)

    convert_filter_output_km_to_m_for_plotting(d, R_km=R_km)
    return run_filter_post_processing(out=d)


def convert_filter_output_km_to_m_for_plotting(d: dict, *, R_km: np.ndarray):
    # This script always uses 6-state LKF outputs in km / km/s.
    # Convert to m / m/s so the shared plotting helpers' axis labels are correct.
    d["xhat_meas"] = np.asarray(d["xhat_meas"], dtype=float) * 1000.0
    d["Xhat_meas"] = np.asarray(d["Xhat_meas"], dtype=float) * 1000.0
    d["X_pf"] = np.asarray(d["X_pf"], dtype=float) * 1000.0
    d["state_error_meas"] = np.asarray(d["state_error_meas"], dtype=float) * 1000.0

    d["P_meas"] = np.asarray(d["P_meas"], dtype=float) * (1000.0 ** 2)
    d["Phat_meas"] = np.asarray(d["Phat_meas"], dtype=float) * (1000.0 ** 2)
    d["P_pf"] = np.asarray(d["P_pf"], dtype=float) * (1000.0 ** 2)

    d["two_sigma_meas"] = np.asarray(d["two_sigma_meas"], dtype=float) * 1000.0

    # Residuals are [range, range-rate] in km and km/s.
    d["prefit_resids_final"] = np.asarray(d["prefit_resids_final"], dtype=float) * 1000.0
    d["postfit_resids_linear_final"] = np.asarray(d["postfit_resids_linear_final"], dtype=float) * 1000.0
    d["postfit_resids_meas"] = np.asarray(d["postfit_resids_meas"], dtype=float) * 1000.0

    # Measurement covariance is diag([km^2, (km/s)^2]); same 1000^2 scale on both entries.
    d["R"] = np.asarray(R_km, dtype=float) * (1000.0 ** 2)


def save_selected_plots(result, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    make_state_errors_eci_plots(result, outdir)
    make_postfit_residuals_linear_plot(result, outdir)


def print_state_error_summary(label: str, result):
    if result.state_error_meas is None:
        print(f"\n{label}: no truth-aligned state error available.")
        return

    e = np.asarray(result.state_error_meas, dtype=float)
    pos_rms = float(np.sqrt(np.nanmean(np.sum(e[:, 0:3] ** 2, axis=1))))
    vel_rms = float(np.sqrt(np.nanmean(np.sum(e[:, 3:6] ** 2, axis=1))))
    print(f"\n{label} state error RMS:")
    print(f"  |dr| RMS = {pos_rms:.6e}")
    print(f"  |dv| RMS = {vel_rms:.6e}")


def print_component_rms_comparison_table(result_no, result_sm):
    e_no = np.asarray(result_no.state_error_meas, dtype=float)
    e_sm = np.asarray(result_sm.state_error_meas, dtype=float)
    if e_no.shape != e_sm.shape:
        raise RuntimeError("Filtered and smoothed state-error arrays do not have matching shapes.")

    labels = ["x", "y", "z", "vx", "vy", "vz"]
    units = ["m", "m", "m", "m/s", "m/s", "m/s"]

    rms_no = np.sqrt(np.nanmean(e_no**2, axis=0))
    rms_sm = np.sqrt(np.nanmean(e_sm**2, axis=0))
    delta = rms_sm - rms_no
    pct = np.full(6, np.nan, dtype=float)
    nz = np.abs(rms_no) > 0.0
    pct[nz] = 100.0 * delta[nz] / rms_no[nz]

    print("\nComponent RMS Comparison (complete dataset)")
    print("  LKF filtered vs RTS smoothed")
    print(f"  {'Comp':<4s} {'Unit':<4s} {'RMS no smooth':>16s} {'RMS smooth':>16s} {'Delta':>14s} {'Delta %':>10s}")
    for i in range(6):
        print(
            f"  {labels[i]:<4s} {units[i]:<4s} "
            f"{rms_no[i]:16.6e} {rms_sm[i]:16.6e} {delta[i]:14.6e} {pct[i]:10.3f}"
        )


def main():
    all_meas, Xtrue_meas, stations, R, P0, x0_bar = load_problem2_inputs()

    mu = 398600.4415
    J2 = 0.0010826269
    J3 = -2.5324e-6

    out_no_smoother = run_lkf_case(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0_bar=x0_bar,
        P0=P0,
        R=R,
        mu=mu,
        J2=J2,
        J3=J3,
        run_smoother=False,
    )

    out_with_smoother = run_lkf_case(
        all_meas=all_meas,
        stations=stations,
        Xtrue_meas=Xtrue_meas,
        x0_bar=x0_bar,
        P0=P0,
        R=R,
        mu=mu,
        J2=J2,
        J3=J3,
        run_smoother=True,
        smooth_back_points=None,
    )

    result_no = to_plotting_result(out_no_smoother, use_smoothed_state=False, R_km=R)
    result_sm = to_plotting_result(out_with_smoother, use_smoothed_state=True, R_km=R)

    plot_root = Path(__file__).resolve().parent / "Plots" / "Smoother_Test"
    plot_dir_no = plot_root / "NoSmoother"
    plot_dir_sm = plot_root / "WithSmoother"

    save_selected_plots(result_no, plot_dir_no)
    save_selected_plots(result_sm, plot_dir_sm)

    print("\nSmoother Test (LKF, Q=0):")
    print("  Saved plots for no-smoother and RTS-smoothed runs.")
    print("  Note: linearized post-fit residual plots are expected to match,")
    print("  because RTS smoothing is a backward post-process and does not change")
    print("  the forward LKF measurement-update residuals.")

    print_state_error_summary("No smoother (filtered)", result_no)
    print_state_error_summary("With smoother (RTS)", result_sm)
    print_component_rms_comparison_table(result_no, result_sm)

    print_rms_summary(result_no, ignore_first_pass=False)
    print_rms_summary(result_sm, ignore_first_pass=False)

    print(f"\nNo-smoother plots: {plot_dir_no}")
    print(f"With-smoother plots: {plot_dir_sm}")


if __name__ == "__main__":
    main()
