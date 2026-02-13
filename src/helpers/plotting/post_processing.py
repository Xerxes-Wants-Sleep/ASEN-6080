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

    # measurement covariance (optional, for normalized RMS)
    R: np.ndarray | None = None

    # computed here (if truth available)
    pos_err_rsw: np.ndarray | None = None         # (m,3)
    Pdiag_pos_rsw: np.ndarray | None = None       # (m,3)  (variance diag in RSW for position)


def compute_rsw_errors_and_cov(
    xhat_meas: np.ndarray,
    P_meas: np.ndarray | None,
    state_error_meas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:

    m = xhat_meas.shape[0]
    pos_err_rsw = np.full((m, 3), np.nan, dtype=float)

    # If covariance history isn't provided, return None for Pdiag_pos_rsw
    Pdiag_pos_rsw = None if P_meas is None else np.full((m, 3), np.nan, dtype=float)

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
        ECI2RSW = np.vstack((Rhat, Shat, What))

        pos_err_rsw[i, :] = (ECI2RSW @ pos_err[i, :].reshape(3, 1)).ravel()

        if P_meas is not None:
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

    Rinv = None
    if result.R is not None:
        Rinv = np.linalg.inv(np.asarray(result.R, dtype=float))

    def rms_norm(resid: np.ndarray | None) -> float | None:
        if resid is None or Rinv is None:
            return None
        r = np.asarray(resid, dtype=float)
        if r.size == 0:
            return None
        norm_sq = np.einsum("ij,jk,ik->i", r, Rinv, r)
        return float(np.sqrt(np.nanmean(norm_sq)))

    # Prefit (final iter) — aligned arrays
    if result.prefit_resids_final is not None:
        pre = result.prefit_resids_final
        pre_norm = rms_norm(pre[keep, :])
        print(f"\nPre-fit residual RMS {suffix}")
        if pre_norm is None:
            print(f"  RMS rho = {rms_nan(pre[keep, 0]):g} m | RMS rhodot = {rms_nan(pre[keep, 1]):g} m/s")
        else:
            print(
                f"  RMS rho = {rms_nan(pre[keep, 0]):g} m | "
                f"RMS rhodot = {rms_nan(pre[keep, 1]):g} m/s | "
                f"RMS norm = {pre_norm:g}"
            )

    # Linearized postfit (final iter)
    if result.postfit_resids_linear_final is not None:
        pf_lin = result.postfit_resids_linear_final
        pf_lin_norm = rms_norm(pf_lin[keep, :])
        print(f"\nLinearized post-fit residual RMS {suffix}")
        if pf_lin_norm is None:
            print(f"  RMS rho = {rms_nan(pf_lin[keep, 0]):g} m | RMS rhodot = {rms_nan(pf_lin[keep, 1]):g} m/s")
        else:
            print(
                f"  RMS rho = {rms_nan(pf_lin[keep, 0]):g} m | "
                f"RMS rhodot = {rms_nan(pf_lin[keep, 1]):g} m/s | "
                f"RMS norm = {pf_lin_norm:g}"
            )

    # Nonlinear postfit (final estimate propagated)
    pf = result.postfit_resids_meas
    pf_norm = rms_norm(pf[keep, :])
    print(f"\nNonlinear post-fit residual RMS {suffix}")
    if pf_norm is None:
        print(f"  RMS rho = {rms_nan(pf[keep, 0]):g} m | RMS rhodot = {rms_nan(pf[keep, 1]):g} m/s")
    else:
        print(
            f"  RMS rho = {rms_nan(pf[keep, 0]):g} m | "
            f"RMS rhodot = {rms_nan(pf[keep, 1]):g} m/s | "
            f"RMS norm = {pf_norm:g}"
        )

    # State error RMS (if truth provided)
    if result.state_error_meas is not None:
        e = result.state_error_meas

        rms_pos = rms_nan(np.linalg.norm(e[keep, 0:3], axis=1))
        rms_vel = rms_nan(np.linalg.norm(e[keep, 3:6], axis=1))

        print(f"\nState RMS {suffix}")
        print(f"  POS 3-norm = {rms_pos:g} m")
        print(f"  VEL 3-norm = {rms_vel:g} m/s")

        if result.pos_err_rsw is not None:
            rms_rsw = np.array([
                rms_nan(result.pos_err_rsw[keep, 0]),
                rms_nan(result.pos_err_rsw[keep, 1]),
                rms_nan(result.pos_err_rsw[keep, 2]),
            ])
            print(f"\nRSW position component RMS {suffix}")
            print(f"  R = {rms_rsw[0]:g} m, S = {rms_rsw[1]:g} m, W = {rms_rsw[2]:g} m")


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


class Result(dict):
    """Dict that also supports attribute access: r.t_meas <-> r['t_meas']"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value


def as_result(x):
    """Ensure x supports attribute access."""
    if isinstance(x, Result):
        return x
    if isinstance(x, dict):
        return Result(x)
    return x

from dataclasses import fields

def run_filter_post_processing(*, out: dict) -> BatchPostProcessResult:
    """
    Convert an LKF/EKF output dict into BatchPostProcessResult so all the
    batch plotting functions work unchanged.
    """
    d = dict(out)

    # ---- key aliases to match BatchPostProcessResult ----
    if "xhat_meas" not in d and "Xhat_meas" in d:
        d["xhat_meas"] = d["Xhat_meas"]

    if "rms_by_iter" not in d:
        d["rms_by_iter"] = None

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

