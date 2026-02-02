from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

from src.Functions.postprocess_batch_for_plots import postprocess_batch_for_plots
from .common import rms_nan


@dataclass
class BatchPostProcessResult:
    # from postprocess_batch_for_plots
    t_meas: np.ndarray
    station_meas: list[str]
    xhat_meas: np.ndarray
    P_meas: np.ndarray
    two_sigma_meas: np.ndarray
    state_error_meas: np.ndarray | None
    postfit_resids_meas: np.ndarray
    rms_final: dict[str, Any]
    rms_by_iter: dict[str, Any] | None

    # from batch info (optional)
    prefit_resids_final: np.ndarray | None = None
    postfit_resids_linear_final: np.ndarray | None = None

    # computed here (if truth available)
    pos_err_rsw: np.ndarray | None = None         # (m,3)
    Pdiag_pos_rsw: np.ndarray | None = None       # (m,3)  (variance diag in RSW for position)


def compute_rsw_errors_and_cov(
    xhat_meas: np.ndarray,
    P_meas: np.ndarray,
    state_error_meas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match MATLAB logic:
      - Compute RSW frame from truth state (xtrue = xhat - err)
      - pos_err_rsw = ECI2RSW * pos_err_eci
      - Pdiag_pos_rsw = diag( ECI2RSW * P_pos * ECI2RSW^T )
    """
    m = xhat_meas.shape[0]
    pos_err_rsw = np.full((m, 3), np.nan, dtype=float)
    Pdiag_pos_rsw = np.full((m, 3), np.nan, dtype=float)

    # truth reconstructed from estimate and error
    xtrue = xhat_meas - state_error_meas
    rtrue = xtrue[:, 0:3]
    vtrue = xtrue[:, 3:6]
    pos_err = state_error_meas[:, 0:3]

    for i in range(m):
        if not np.all(np.isfinite(rtrue[i])) or not np.all(np.isfinite(vtrue[i])):
            continue

        r = rtrue[i]
        v = vtrue[i]

        rnorm = np.linalg.norm(r)
        h = np.cross(r, v)
        hnorm = np.linalg.norm(h)

        if rnorm < 1e-12 or hnorm < 1e-12:
            continue

        Rhat = r / rnorm
        What = h / hnorm
        Shat = np.cross(What, Rhat)

        ECI2RSW = np.vstack((Rhat, Shat, What))  # 3x3

        pos_err_rsw[i, :] = (ECI2RSW @ pos_err[i, :].reshape(3, 1)).ravel()

        Ppos = P_meas[i, 0:3, 0:3]
        P_rsw = ECI2RSW @ Ppos @ ECI2RSW.T
        Pdiag_pos_rsw[i, :] = np.diag(P_rsw)

    return pos_err_rsw, Pdiag_pos_rsw


def print_rms_summary(result: BatchPostProcessResult, ignore_first_pass: bool = False) -> None:
    """
    Similar spirit to professor prints. Uses NaN-safe RMS.
    """
    t = result.t_meas
    keep_all = np.ones_like(t, dtype=bool)
    keep_ignore = result.rms_final.get("keep_ignore_first_mask", keep_all)

    keep = keep_ignore if ignore_first_pass else keep_all
    suffix = "(ignore first pass)" if ignore_first_pass else "(all meas)"

    # Prefit (final iter) — aligned arrays
    if result.prefit_resids_final is not None:
        pre = result.prefit_resids_final
        print(f"\nPre-fit residual RMS {suffix}")
        print(f"  Rho    = {rms_nan(pre[keep, 0]):g} km")
        print(f"  RhoDot = {rms_nan(pre[keep, 1]):g} km/s")

    # Linearized postfit (final iter)
    if result.postfit_resids_linear_final is not None:
        pf_lin = result.postfit_resids_linear_final
        print(f"\nLinearized post-fit residual RMS {suffix}")
        print(f"  Rho    = {rms_nan(pf_lin[keep, 0]):g} km")
        print(f"  RhoDot = {rms_nan(pf_lin[keep, 1]):g} km/s")

    # Nonlinear postfit (final estimate propagated)
    pf = result.postfit_resids_meas
    print(f"\nNonlinear post-fit residual RMS {suffix}")
    print(f"  Rho    = {rms_nan(pf[keep, 0]):g} km")
    print(f"  RhoDot = {rms_nan(pf[keep, 1]):g} km/s")

    # State error RMS (if truth provided)
    if result.state_error_meas is not None:
        e = result.state_error_meas

        rms_pos = rms_nan(np.linalg.norm(e[keep, 0:3], axis=1))
        rms_vel = rms_nan(np.linalg.norm(e[keep, 3:6], axis=1))

        print(f"\nState RMS {suffix}")
        print(f"  POS 3-norm = {rms_pos:g} km")
        print(f"  VEL 3-norm = {rms_vel:g} km/s")

        if result.pos_err_rsw is not None:
            rms_rsw = np.array([
                rms_nan(result.pos_err_rsw[keep, 0]),
                rms_nan(result.pos_err_rsw[keep, 1]),
                rms_nan(result.pos_err_rsw[keep, 2]),
            ])
            print(f"\nRSW position component RMS {suffix}")
            print(f"  R = {rms_rsw[0]:g} km, S = {rms_rsw[1]:g} km, W = {rms_rsw[2]:g} km")


def run_batch_post_processing(
    *,
    all_meas,
    stations,
    x0_hat: np.ndarray,
    P0_hat: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    truth_times: np.ndarray | None = None,
    truth_states_6: np.ndarray | None = None,
    info: dict[str, Any] | None = None,
    reltol: float = 1e-10,
    abstol: float = 1e-10,
    first_pass_gap_s: float = 6 * 3600.0,
    x0_star_hist: list[np.ndarray] | None = None,
) -> BatchPostProcessResult:

    out = postprocess_batch_for_plots(
        all_meas=all_meas,
        stations=stations,
        x0_hat=x0_hat,
        P0_hat=P0_hat,
        mu=mu,
        J2=J2,
        J3=J3,
        truth_times=truth_times,
        truth_states_6=truth_states_6,
        reltol=reltol,
        abstol=abstol,
        x0_star_hist=x0_star_hist,
        first_pass_gap_s=first_pass_gap_s,
    )

    result = BatchPostProcessResult(**out)

    # attach batch info residuals (final iteration aligned)
    if info is not None:
        result.prefit_resids_final = info.get("prefit_resids_final", None)
        result.postfit_resids_linear_final = info.get("postfit_resids_linear_final", None)

    # compute RSW errors/cov if truth available
    if result.state_error_meas is not None:
        pos_err_rsw, Pdiag_pos_rsw = compute_rsw_errors_and_cov(
            result.xhat_meas, result.P_meas, result.state_error_meas
        )
        result.pos_err_rsw = pos_err_rsw
        result.Pdiag_pos_rsw = Pdiag_pos_rsw

    return result
