import numpy as np
from .batch import propagate_state_and_stm_history

def postprocess_batch_for_plots(
    all_meas,
    stations,
    x0_hat: np.ndarray,
    P0_hat: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    Re: float = 6378.0,
    truth_times: np.ndarray | None = None,
    truth_states_6: np.ndarray | None = None,
    reltol: float = 1e-10,
    abstol: float = 1e-10,
    x0_star_hist: list | None = None,
    first_pass_gap_s: float = 6 * 3600.0,   # heuristic: gap > 6 hours starts a new pass
):
    """
    Post-processing helper to support:
      1) state error vs time with +/-2sigma bounds (if truth provided)
      2) postfit residual plots (also returned here for convenience)

    This propagates the FINAL batch estimate to each measurement time, computes:
      - xhat(ti)
      - Phi(ti,t0)
      - P(ti) = Phi P0_hat Phi^T
      - 2sigma(ti) from diag(P(ti))
      - state_error(ti) = xhat(ti) - xtrue(ti) if truth provided
      - postfit residuals: y_post(ti) = Y(ti) - yhat(xhat(ti))

    Parameters
    ----------
    all_meas : list[dict]
        Measurements (noisy), with keys: station, t, rho_km, rho_dot_km_s
    stations : list[Stations]
        Station objects
    x0_hat : (6,)
        Batch-estimated initial state
    P0_hat : (6,6)
        Batch-estimated covariance at t0
    truth_times : (N,) optional
        Times corresponding to truth_states_6
    truth_states_6 : (N,6) optional
        Truth trajectory [r;v] at truth_times
        If provided, state errors will be computed by interpolating truth onto measurement times.
        (Interpolation is component-wise linear.)
    Returns
    -------
    out : dict with arrays:
        t_meas               (m,)
        station_meas         list length m
        xhat_meas            (m,6)
        P_meas               (m,6,6)
        two_sigma_meas       (m,6)
        state_error_meas     (m,6) or None
        postfit_resids_meas  (m,2)

        rms_final            dict with RMS (all meas) + (ignore first pass)
        rms_by_iter          dict with arrays vs iteration (if x0_star_hist provided)
    """
    # Sort measurements and station lookup
    station_map = {st.name: st for st in stations}

    # ensure time-sorted measurements (important for the "first pass" logic)
    all_meas = sorted(all_meas, key=lambda m: float(m["t"]))

    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
    st_meas = [m["station"] for m in all_meas]

    # Reference epoch for STM mapping
    t0 = float(all_meas[0]["t"])

    mcount = len(all_meas)
    xhat_meas = np.zeros((mcount, 6), dtype=float)
    P_meas = np.zeros((mcount, 6, 6), dtype=float)
    two_sigma = np.zeros((mcount, 6), dtype=float)

    # Allow elevation masking: keep fixed-size arrays, but store NaN where no measurement is available
    postfit_resids = np.full((mcount, 2), np.nan, dtype=float)

    # If truth provided, interpolate truth to measurement times (component-wise)
    state_error = None
    xtrue_interp = None
    if truth_times is not None and truth_states_6 is not None:
        truth_times = np.asarray(truth_times, dtype=float).reshape(-1,)
        truth_states_6 = np.asarray(truth_states_6, dtype=float)
        if truth_states_6.shape[1] != 6:
            raise ValueError("truth_states_6 must be shape (N,6)")

        xtrue_interp = np.zeros((mcount, 6), dtype=float)
        for j in range(6):
            xtrue_interp[:, j] = np.interp(t_meas, truth_times, truth_states_6[:, j])
        state_error = np.zeros((mcount, 6), dtype=float)

    # --- helper: measurement prediction using Stations.measure (supports elevation mask) ---
    def yhat_from_station(st, x6, ti):
        d = st.measure(x6[:3], x6[3:], float(ti))
        if d is None:
            return None
        return np.array([d["rho_km"], d["rho_dot_km_s"]], dtype=float)

    # --- propagate final estimate to all measurement times (fast) ---
    X_hist, Phi_hist = propagate_state_and_stm_history(
        x0_ref_6=x0_hat,
        t_eval=t_meas,
        mu=mu, J2=J2, J3=J3,
        Re=Re,
        reltol=reltol,
        abstol=abstol,
    )

    # Main loop over measurement times
    for i, m in enumerate(all_meas):
        ti = float(m["t"])
        st = station_map[m["station"]]

        x_i = X_hist[i, :]
        Phi_i0 = Phi_hist[i, :, :]

        xhat_meas[i, :] = x_i

        # Covariance along the trajectory: P(ti) = Phi P0 Phi^T
        Pi = Phi_i0 @ P0_hat @ Phi_i0.T
        P_meas[i, :, :] = Pi

        # +/- 2 sigma bounds from diagonal
        two_sigma[i, :] = 2.0 * np.sqrt(np.maximum(np.diag(Pi), 0.0))

        # State error if truth provided
        if state_error is not None and xtrue_interp is not None:
            state_error[i, :] = x_i - xtrue_interp[i, :]

        # Postfit residuals at measurement times
        Yi = np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)
        yhat_i = yhat_from_station(st, x_i, ti)
        if yhat_i is None:
            continue
        postfit_resids[i, :] = Yi - yhat_i

    # -----------------------------
    # RMS COMPUTATION HELPERS
    # -----------------------------
    def rms_nan(A, axis=0):
        """RMS ignoring NaNs."""
        return np.sqrt(np.nanmean(A**2, axis=axis))

    def compute_first_pass_mask(t, gap_s):
        """
        Define 'first pass' as measurements before the first large time gap.
        If no gap found, just use first measurement as the divider.
        """
        if t.size < 2:
            return np.ones_like(t, dtype=bool), np.zeros_like(t, dtype=bool)

        dt = np.diff(t)
        idx_gap = np.where(dt > gap_s)[0]
        
        keep_all = np.ones_like(t, dtype=bool)
        keep_ignore_first = np.ones_like(t, dtype=bool)
        
        if idx_gap.size == 0:
            # No big gap found -> just exclude the first measurement only
            keep_ignore_first[0] = False
        else:
            # Gap found -> exclude everything up to and including the first gap
            end_first_pass = idx_gap[0]
            keep_ignore_first[: end_first_pass + 1] = False
        
        return keep_all, keep_ignore_first
    
    keep_all, keep_ignore_first = compute_first_pass_mask(t_meas, first_pass_gap_s)

    # RMS: state errors (if truth available)
    rms_state_all = None
    rms_state_ignore = None
    rms_pos3_all = None
    rms_pos3_ignore = None
    rms_vel3_all = None
    rms_vel3_ignore = None

    if state_error is not None:
        e = state_error
        rms_state_all = rms_nan(e[keep_all, :], axis=0)
        rms_state_ignore = rms_nan(e[keep_ignore_first, :], axis=0)

        rms_pos3_all = float(rms_nan(np.linalg.norm(e[keep_all, 0:3], axis=1), axis=0))
        rms_pos3_ignore = float(rms_nan(np.linalg.norm(e[keep_ignore_first, 0:3], axis=1), axis=0))

        rms_vel3_all = float(rms_nan(np.linalg.norm(e[keep_all, 3:6], axis=1), axis=0))
        rms_vel3_ignore = float(rms_nan(np.linalg.norm(e[keep_ignore_first, 3:6], axis=1), axis=0))

    # RMS: postfit residuals (works even without truth)
    r = postfit_resids
    rms_post_all = rms_nan(r[keep_all, :], axis=0)
    rms_post_ignore = rms_nan(r[keep_ignore_first, :], axis=0)

    rms_final = {
        "keep_all_mask": keep_all,
        "keep_ignore_first_mask": keep_ignore_first,
        "state_comp_all": rms_state_all,                 # (6,) or None
        "state_comp_ignore_first": rms_state_ignore,     # (6,) or None
        "pos3_all": rms_pos3_all,                        # float or None
        "pos3_ignore_first": rms_pos3_ignore,            # float or None
        "vel3_all": rms_vel3_all,                        # float or None
        "vel3_ignore_first": rms_vel3_ignore,            # float or None
        "postfit_all": rms_post_all,                     # (2,)
        "postfit_ignore_first": rms_post_ignore,         # (2,)
    }

    # -----------------------------
    # RMS VS ITERATION (batch only)
    # -----------------------------
    rms_by_iter = None
    if x0_star_hist is not None and len(x0_star_hist) > 0:
        # allocate: one RMS per iteration (including iter 0 = initial guess)
        K = len(x0_star_hist)

        state_comp_all_hist = None
        state_comp_ign_hist = None
        pos3_all_hist = None
        pos3_ign_hist = None
        vel3_all_hist = None
        vel3_ign_hist = None

        if state_error is not None:
            state_comp_all_hist = np.full((K, 6), np.nan, dtype=float)
            state_comp_ign_hist = np.full((K, 6), np.nan, dtype=float)
            pos3_all_hist = np.full((K,), np.nan, dtype=float)
            pos3_ign_hist = np.full((K,), np.nan, dtype=float)
            vel3_all_hist = np.full((K,), np.nan, dtype=float)
            vel3_ign_hist = np.full((K,), np.nan, dtype=float)

        post_all_hist = np.full((K, 2), np.nan, dtype=float)
        post_ign_hist = np.full((K, 2), np.nan, dtype=float)

        for k in range(K):
            x0k = np.asarray(x0_star_hist[k], dtype=float).reshape(6,)

            Xk, _ = propagate_state_and_stm_history(
                x0_ref_6=x0k,
                t_eval=t_meas,
                mu=mu, J2=J2, J3=J3,
                Re=Re,
                reltol=reltol,
                abstol=abstol,
            )

            # residuals for this iteration
            rk = np.full((mcount, 2), np.nan, dtype=float)
            ek = None
            if xtrue_interp is not None:
                ek = Xk - xtrue_interp  # (m,6)

            for i, m in enumerate(all_meas):
                ti = float(m["t"])
                st = station_map[m["station"]]
                Yi = np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)
                yhat_i = yhat_from_station(st, Xk[i, :], ti)
                if yhat_i is None:
                    continue
                rk[i, :] = Yi - yhat_i

            # fill RMS histories
            post_all_hist[k, :] = rms_nan(rk[keep_all, :], axis=0)
            post_ign_hist[k, :] = rms_nan(rk[keep_ignore_first, :], axis=0)

            if ek is not None:
                state_comp_all_hist[k, :] = rms_nan(ek[keep_all, :], axis=0)
                state_comp_ign_hist[k, :] = rms_nan(ek[keep_ignore_first, :], axis=0)

                pos3_all_hist[k] = float(rms_nan(np.linalg.norm(ek[keep_all, 0:3], axis=1), axis=0))
                pos3_ign_hist[k] = float(rms_nan(np.linalg.norm(ek[keep_ignore_first, 0:3], axis=1), axis=0))

                vel3_all_hist[k] = float(rms_nan(np.linalg.norm(ek[keep_all, 3:6], axis=1), axis=0))
                vel3_ign_hist[k] = float(rms_nan(np.linalg.norm(ek[keep_ignore_first, 3:6], axis=1), axis=0))

        rms_by_iter = {
            "state_comp_all": state_comp_all_hist,                 # (K,6) or None
            "state_comp_ignore_first": state_comp_ign_hist,         # (K,6) or None
            "pos3_all": pos3_all_hist,                               # (K,) or None
            "pos3_ignore_first": pos3_ign_hist,                      # (K,) or None
            "vel3_all": vel3_all_hist,                               # (K,) or None
            "vel3_ignore_first": vel3_ign_hist,                      # (K,) or None
            "postfit_all": post_all_hist,                            # (K,2)
            "postfit_ignore_first": post_ign_hist,                   # (K,2)
        }

    return {
        "t_meas": t_meas,
        "station_meas": st_meas,
        "xhat_meas": xhat_meas,
        "P_meas": P_meas,
        "two_sigma_meas": two_sigma,
        "state_error_meas": state_error,
        "postfit_resids_meas": postfit_resids,
        "rms_final": rms_final,
        "rms_by_iter": rms_by_iter,
    }
