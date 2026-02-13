from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np

from postprocess_batch18_for_plots import postprocess_batch18_for_plots
from src.helpers.plotting.post_processing import BatchPostProcessResult, compute_rsw_errors_and_cov


def run_batch_post_processing_18(
    *,
    all_meas,
    x0_hat: np.ndarray,
    P0_hat: np.ndarray,
    info: dict[str, Any],
    truth_times: np.ndarray | None = None,
    truth_states_6: np.ndarray | None = None,
    length_unit_in: str = "m",
    length_unit_out: str = "m",
    truth_length_unit: str | None = None,
    first_pass_gap_s: float = 6 * 3600.0,
) -> BatchPostProcessResult:

    out = postprocess_batch18_for_plots(
        all_meas=all_meas,
        x0_hat=x0_hat,
        P0_hat=P0_hat,
        info=info,
        truth_times=truth_times,
        truth_states_6=truth_states_6,
        length_unit_in=length_unit_in,
        length_unit_out=length_unit_out,
        truth_length_unit=truth_length_unit,
        first_pass_gap_s=first_pass_gap_s,
    )

    result = BatchPostProcessResult(**out)

    # Attach batch info residuals (final iteration aligned)
    def _pick(info_dict, *keys):
        for k in keys:
            if k in info_dict and info_dict[k] is not None:
                return info_dict[k]
        return None

    def _length_scale(unit_in: str, unit_out: str) -> float:
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
        if u_in not in to_m or u_out not in to_m:
            raise ValueError(f"Unknown length unit conversion: {unit_in} -> {unit_out}")
        return to_m[u_in] / to_m[u_out]

    scale = _length_scale(length_unit_in, length_unit_out)

    prefit_raw = _pick(info, "prefit_resids_final", "prefit_residuals")
    if prefit_raw is not None:
        result.prefit_resids_final = np.asarray(prefit_raw, dtype=float) * scale

    # Linear postfit may not exist for 18-state; fallback to nonlinear postfit
    postfit_lin_raw = _pick(
        info,
        "postfit_resids_linear_final",
        "postfit_residuals",
        "postfit_resids_meas",
    )
    if postfit_lin_raw is not None:
        result.postfit_resids_linear_final = np.asarray(postfit_lin_raw, dtype=float) * scale

    # Compute RSW errors/cov if truth available
    if result.state_error_meas is not None:
        pos_err_rsw, Pdiag_pos_rsw = compute_rsw_errors_and_cov(
            result.xhat_meas, result.P_meas, result.state_error_meas
        )
        result.pos_err_rsw = pos_err_rsw
        result.Pdiag_pos_rsw = Pdiag_pos_rsw

    return result


def run_filter_post_processing_18(
    *,
    out: dict,
    length_unit_in: str = "m",
    length_unit_out: str = "m",
) -> BatchPostProcessResult:
    """
    Convert 18-state filter output into BatchPostProcessResult for plotting.
    Slices to the first 6 states and converts units (default m -> m).
    """

    def _length_scale(unit_in: str, unit_out: str) -> float:
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
        if u_in not in to_m or u_out not in to_m:
            raise ValueError(f"Unknown length unit conversion: {unit_in} -> {unit_out}")
        return to_m[u_in] / to_m[u_out]

    scale = _length_scale(length_unit_in, length_unit_out)

    d = dict(out)

    # --- aliases ---
    if "xhat_meas" not in d and "Xhat_meas" in d:
        d["xhat_meas"] = d["Xhat_meas"]
    if "P_meas" not in d and "Phat_meas" in d:
        d["P_meas"] = d["Phat_meas"]

    # --- slice + scale ---
    if d.get("xhat_meas", None) is not None:
        xh = np.asarray(d["xhat_meas"], dtype=float)
        if xh.shape[1] >= 6:
            d["xhat_meas"] = xh[:, 0:6] * scale

    if d.get("P_meas", None) is not None:
        Pm = np.asarray(d["P_meas"], dtype=float)
        if Pm.shape[1] >= 6 and Pm.shape[2] >= 6:
            d["P_meas"] = Pm[:, 0:6, 0:6] * (scale ** 2)

    if d.get("two_sigma_meas", None) is not None:
        ts = np.asarray(d["two_sigma_meas"], dtype=float)
        if ts.shape[1] >= 6:
            d["two_sigma_meas"] = ts[:, 0:6] * scale
        else:
            d["two_sigma_meas"] = ts * scale

    if d.get("state_error_meas", None) is not None:
        e = np.asarray(d["state_error_meas"], dtype=float)
        if e.shape[1] >= 6:
            d["state_error_meas"] = e[:, 0:6] * scale

    # residuals are in length units
    for k in ("prefit_resids_final", "postfit_resids_linear_final", "postfit_resids_meas"):
        if d.get(k, None) is not None:
            d[k] = np.asarray(d[k], dtype=float) * scale

    # measurement covariance (rho, rhodot) scales with length^2
    if d.get("R", None) is not None:
        d["R"] = np.asarray(d["R"], dtype=float) * (scale ** 2)

    # P_pf is flattened covariance; scale if present
    if d.get("P_pf", None) is not None:
        d["P_pf"] = np.asarray(d["P_pf"], dtype=float) * (scale ** 2)

    # ---- only pass dataclass fields (ignore extra keys) ----
    allowed = {f.name for f in fields(BatchPostProcessResult)}
    payload = {k: d.get(k, None) for k in allowed}

    result = BatchPostProcessResult(**payload)

    # ---- compute RSW fields if truth available ----
    if result.state_error_meas is not None:
        pos_err_rsw, Pdiag_pos_rsw = compute_rsw_errors_and_cov(
            result.xhat_meas, result.P_meas, result.state_error_meas
        )
        result.pos_err_rsw = pos_err_rsw
        result.Pdiag_pos_rsw = Pdiag_pos_rsw

    return result
