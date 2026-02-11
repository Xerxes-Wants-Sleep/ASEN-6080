import numpy as np


def _length_scale(unit_in: str, unit_out: str) -> float:
    """
    Return multiplicative factor to convert length-based units.
    Example: unit_in="m", unit_out="km" -> 0.001
    """
    u_in = str(unit_in).strip().lower()
    u_out = str(unit_out).strip().lower()

    to_m = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "km": 1000.0,
        "kilometer": 1000.0,
        "kilometers": 1000.0,
    }

    if u_in not in to_m:
        raise ValueError(f"Unknown length unit: {unit_in}")
    if u_out not in to_m:
        raise ValueError(f"Unknown length unit: {unit_out}")

    return to_m[u_in] / to_m[u_out]


def _rms_nan(A, axis=0):
    """RMS ignoring NaNs."""
    return np.sqrt(np.nanmean(A**2, axis=axis))


def _compute_first_pass_mask(t, gap_s):
    """
    Define "first pass" as measurements before the first large time gap.
    If no gap found, just exclude the first measurement only.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size < 2:
        return np.ones_like(t, dtype=bool), np.zeros_like(t, dtype=bool)

    dt = np.diff(t)
    idx_gap = np.where(dt > gap_s)[0]

    keep_all = np.ones_like(t, dtype=bool)
    keep_ignore_first = np.ones_like(t, dtype=bool)

    if idx_gap.size == 0:
        keep_ignore_first[0] = False
    else:
        end_first_pass = idx_gap[0]
        keep_ignore_first[: end_first_pass + 1] = False

    return keep_all, keep_ignore_first


def postprocess_batch18_for_plots(
    *,
    all_meas,
    x0_hat: np.ndarray,
    P0_hat: np.ndarray,
    info: dict,
    truth_times: np.ndarray | None = None,
    truth_states_6: np.ndarray | None = None,
    length_unit_in: str = "m",
    length_unit_out: str = "m",
    truth_length_unit: str | None = None,
    first_pass_gap_s: float = 6 * 3600.0,
):
    """
    Post-process helper for batch_18_state output. Returns a dict compatible
    with the plotting pipeline (BatchPostProcessResult).

    Notes
    -----
    - Uses info["state_hist"] and info["P_hist"] from batch_18_state.
    - Only the first 6 states (r,v) are used for plots.
    - Converts units from length_unit_in -> length_unit_out (default m -> m).
    - If truth is provided, it is interpolated onto measurement times.
    """

    if info is None:
        raise ValueError("info dict from batch_18_state is required.")

    # Prefer the sorted times from info to keep alignment.
    if "t_meas" in info and info["t_meas"] is not None:
        t_meas = np.asarray(info["t_meas"], dtype=float).reshape(-1)
    else:
        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
        t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)

    if "station_meas" in info and info["station_meas"] is not None:
        st_meas = list(info["station_meas"])
    else:
        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
        st_meas = [m["station"] for m in all_meas]

    X_hist = info.get("state_hist", None)
    if X_hist is None:
        raise ValueError("info['state_hist'] is required for 18-state post-processing.")
    X_hist = np.asarray(X_hist, dtype=float)
    if X_hist.ndim != 2 or X_hist.shape[1] < 6:
        raise ValueError("info['state_hist'] must have shape (m, n) with n >= 6.")

    P_hist = info.get("P_hist", None)
    if P_hist is None:
        raise ValueError("info['P_hist'] is required for covariance plots.")
    P_hist = np.asarray(P_hist, dtype=float)
    if P_hist.ndim != 3 or P_hist.shape[1] < 6 or P_hist.shape[2] < 6:
        raise ValueError("info['P_hist'] must have shape (m, n, n) with n >= 6.")

    # Slice to spacecraft state only (r,v)
    xhat_meas = X_hist[:, 0:6]
    P_meas = P_hist[:, 0:6, 0:6]

    # Unit conversion
    scale = _length_scale(length_unit_in, length_unit_out)
    xhat_meas = xhat_meas * scale
    P_meas = P_meas * (scale ** 2)

    # +/- 2 sigma bounds
    two_sigma = 2.0 * np.sqrt(np.maximum(np.diagonal(P_meas, axis1=1, axis2=2), 0.0))

    # Postfit residuals (if provided)
    postfit_raw = info.get("postfit_residuals", info.get("postfit_resids_meas", None))
    if postfit_raw is None:
        postfit_resids = np.full((t_meas.size, 2), np.nan, dtype=float)
    else:
        postfit_resids = np.asarray(postfit_raw, dtype=float)
        if postfit_resids.shape[0] != t_meas.size:
            raise ValueError("postfit_residuals length does not match t_meas.")
        postfit_resids = postfit_resids * scale

    # Truth interpolation (component-wise)
    state_error = None
    if truth_times is not None and truth_states_6 is not None:
        truth_times = np.asarray(truth_times, dtype=float).reshape(-1)
        truth_states_6 = np.asarray(truth_states_6, dtype=float)
        if truth_states_6.shape[1] != 6:
            raise ValueError("truth_states_6 must be shape (N,6)")

        truth_unit = truth_length_unit or length_unit_out
        truth_scale = _length_scale(truth_unit, length_unit_out)
        truth_states_6 = truth_states_6 * truth_scale

        xtrue_interp = np.zeros((t_meas.size, 6), dtype=float)
        for j in range(6):
            xtrue_interp[:, j] = np.interp(t_meas, truth_times, truth_states_6[:, j])

        state_error = xhat_meas - xtrue_interp

    # RMS summary (NaN-safe)
    keep_all, keep_ignore_first = _compute_first_pass_mask(t_meas, first_pass_gap_s)

    rms_state_all = None
    rms_state_ignore = None
    rms_pos3_all = None
    rms_pos3_ignore = None
    rms_vel3_all = None
    rms_vel3_ignore = None

    if state_error is not None:
        e = state_error
        rms_state_all = _rms_nan(e[keep_all, :], axis=0)
        rms_state_ignore = _rms_nan(e[keep_ignore_first, :], axis=0)

        rms_pos3_all = float(_rms_nan(np.linalg.norm(e[keep_all, 0:3], axis=1), axis=0))
        rms_pos3_ignore = float(_rms_nan(np.linalg.norm(e[keep_ignore_first, 0:3], axis=1), axis=0))

        rms_vel3_all = float(_rms_nan(np.linalg.norm(e[keep_all, 3:6], axis=1), axis=0))
        rms_vel3_ignore = float(_rms_nan(np.linalg.norm(e[keep_ignore_first, 3:6], axis=1), axis=0))

    rms_post_all = _rms_nan(postfit_resids[keep_all, :], axis=0)
    rms_post_ignore = _rms_nan(postfit_resids[keep_ignore_first, :], axis=0)

    rms_final = {
        "keep_all_mask": keep_all,
        "keep_ignore_first_mask": keep_ignore_first,
        "state_comp_all": rms_state_all,
        "state_comp_ignore_first": rms_state_ignore,
        "pos3_all": rms_pos3_all,
        "pos3_ignore_first": rms_pos3_ignore,
        "vel3_all": rms_vel3_all,
        "vel3_ignore_first": rms_vel3_ignore,
        "postfit_all": rms_post_all,
        "postfit_ignore_first": rms_post_ignore,
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
        "rms_by_iter": None,
    }
