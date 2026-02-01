############### DOES NOT WORK ###################






import numpy as np
from .filters import ExtendedKalmanFilter, LinearizedKalmanFilter, KalmanFilterBase


def warmstart_ekf_with_lkf(
    *,
    all_meas,
    stations,
    Xtrue_meas,
    N_boot: int,
    X0_star, P0, R, Q, mu, J2, J3,
    reltol=1e-10, abstol=1e-10, method="DOP853", j2=True, j3=False,
    first_pass_gap_s=6*3600.0,
):
    N = len(all_meas)
    if N_boot < 1 or N_boot >= N:
        raise ValueError(f"N_boot must be in [1, {N-1}] but got {N_boot}")

    meas_boot = all_meas[:N_boot]
    Xtrue_boot = None if Xtrue_meas is None else Xtrue_meas[:N_boot, :]

    lkf = LinearizedKalmanFilter(
        X0_star=X0_star, P0=P0, R=R, Q=Q, mu=mu, J2=J2, J3=J3,
        reltol=reltol, abstol=abstol, method=method, j2=j2, j3=j3,
        first_pass_gap_s=first_pass_gap_s
    )
    out_lkf = lkf.run(all_meas=meas_boot, stations=stations, Xtrue_meas=Xtrue_boot)

    x0_ekf = lkf.Xhat.copy()
    P0_ekf = lkf.Phat.copy()

        # posterior at end of bootstrap
    x0_ekf = out_lkf["Xhat_meas"][-1].copy()   # full state, already Xstar + xhat
    P0_ekf = lkf.Phat.copy()                  # covariance of the estimate (same for error vs full since Xstar is deterministic)
    t_prev_init = float(out_lkf["t_meas"][-1])


    # --- covariance conditioning / inflation for EKF handoff ---
    cov_inflate = 100.0          # try 10, 100, 1000
    P0_ekf *= cov_inflate

    # enforce symmetry
    P0_ekf = 0.5 * (P0_ekf + P0_ekf.T)

    # floor the diagonal so it is not unrealistically small (use your original P0 as a floor)
    diag_floor = np.diag(P0)
    P0_ekf += np.diag(np.maximum(diag_floor - np.diag(P0_ekf), 0.0))

    # tiny nugget for numerical safety
    P0_ekf += 1e-18 * np.eye(6)


    meas_rest = all_meas[N_boot:]
    Xtrue_rest = None if Xtrue_meas is None else Xtrue_meas[N_boot:, :]

    ekf = ExtendedKalmanFilter(
        x0=x0_ekf, P0=P0_ekf, R=R, Q=Q, mu=mu, J2=J2, J3=J3,
        reltol=reltol, abstol=abstol, method=method, j2=j2, j3=j3,
        first_pass_gap_s=first_pass_gap_s
    )
    t_prev_init = float(out_lkf["t_meas"][-1])

    out_ekf = ekf.run(
        all_meas=meas_rest,
        stations=stations,
        Xtrue_meas=Xtrue_rest,
        t_prev_init=t_prev_init
    )

    out = {}
    out["t_meas"] = np.hstack([out_lkf["t_meas"], out_ekf["t_meas"]])
    out["station_meas"] = out_lkf["station_meas"] + out_ekf["station_meas"]
    out["Xhat_meas"] = np.vstack([out_lkf["Xhat_meas"], out_ekf["Xhat_meas"]])

    if Xtrue_meas is None:
        out["state_error_meas"] = None
    else:
        out["state_error_meas"] = np.vstack([out_lkf["state_error_meas"], out_ekf["state_error_meas"]])

    out["postfit_resids_meas"] = np.vstack([out_lkf["postfit_resids_meas"], out_ekf["postfit_resids_meas"]])
    out["two_sigma_meas"] = np.vstack([out_lkf["two_sigma_meas"], out_ekf["two_sigma_meas"]])

    out["out_lkf_bootstrap"] = out_lkf
    out["out_ekf_rest"] = out_ekf

    out["rms_final"] = KalmanFilterBase.compute_rms_summary_from_arrays(
        out["t_meas"],
        out["postfit_resids_meas"],
        out["state_error_meas"],
        first_pass_gap_s=first_pass_gap_s
    )

    return out
