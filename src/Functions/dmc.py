from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from .jacobians import accel_wJ2J3, dadr_wJ2J3
from .propagation import PropSettings
from .range_rangerate import H_range_rangerate
from .filters import KalmanFilterBase


def _measurement_model(station, X6: np.ndarray, t: float) -> np.ndarray | None:
    d = station.measure(X6[:3], X6[3:], float(t))
    if d is None:
        return None
    rho = d["rho_km"] if "rho_km" in d else d["rho"]
    rhod = d["rho_dot_km_s"] if "rho_dot_km_s" in d else d["rho_dot"]
    return np.array([rho, rhod], dtype=float)


def _dmc_ref_dynamics(
    t: float,
    x9: np.ndarray,
    *,
    mu: float,
    J2: float,
    J3: float,
    B: np.ndarray,
    Re: float,
    j2: bool,
    j3: bool,
) -> np.ndarray:
    x9 = np.asarray(x9, dtype=float).reshape(9,)
    r = x9[0:3]
    v = x9[3:6]
    w = x9[6:9]

    a = accel_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3) + w
    wdot = -B @ w
    return np.hstack((v, a, wdot))


def _dmc_A_matrix(
    x9: np.ndarray,
    *,
    mu: float,
    J2: float,
    J3: float,
    B: np.ndarray,
    Re: float,
    j2: bool,
    j3: bool,
) -> np.ndarray:
    x9 = np.asarray(x9, dtype=float).reshape(9,)
    r = x9[0:3]

    G = dadr_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)
    A = np.zeros((9, 9), dtype=float)
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = G
    A[3:6, 6:9] = np.eye(3)
    A[6:9, 6:9] = -B
    return A


def _van_loan_discretization(A: np.ndarray, L: np.ndarray, Qc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    if dt <= 0.0:
        return np.eye(n, dtype=float), np.zeros((n, n), dtype=float)

    GQGt = L @ Qc @ L.T
    Z = np.zeros((n, n), dtype=float)
    M = np.block([[-A, GQGt], [Z, A.T]]) * float(dt)
    Mexp = expm(M)

    Phi = Mexp[n:, n:].T
    Phi_inv_Qk = Mexp[0:n, n:]
    Qk = Phi @ Phi_inv_Qk
    Qk = 0.5 * (Qk + Qk.T)
    return Phi, Qk


def _propagate_reference_history(*, x0: np.ndarray, t_eval: np.ndarray, rhs, settings: PropSettings):
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    if t_eval.size == 0:
        return np.empty((0, x0.size), dtype=float), None

    t0 = float(t_eval[0])
    tf = float(t_eval[-1])
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    sol = solve_ivp(
        rhs,
        (t0, tf),
        x0,
        t_eval=t_eval,
        dense_output=True,
        rtol=settings.rtol,
        atol=settings.atol,
        method=settings.method,
    )
    if not sol.success:
        raise RuntimeError(f"DMC reference propagation failed: {sol.message}")
    return sol.y.T, sol.sol


def _propagate_state_step(*, x0: np.ndarray, t0: float, t1: float, rhs, settings: PropSettings) -> np.ndarray:
    if t1 <= t0:
        return np.asarray(x0, dtype=float).reshape(-1).copy()

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    sol = solve_ivp(
        rhs,
        (float(t0), float(t1)),
        x0,
        t_eval=[float(t1)],
        rtol=settings.rtol,
        atol=settings.atol,
        method=settings.method,
    )
    if not sol.success:
        raise RuntimeError(f"DMC state propagation failed: {sol.message}")
    return np.asarray(sol.y[:, -1], dtype=float).reshape(-1)


def lkf_with_dmc(
    *,
    all_meas,
    stations,
    X0_star: np.ndarray,
    P0: np.ndarray,
    R: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    tau_s: float,
    sigma_accel_m_s2: float,
    Xtrue_meas: np.ndarray | None = None,
    Re: float = 6378.0,
    reltol: float = 1.0e-10,
    abstol: float = 1.0e-10,
    method: str = "DOP853",
    j2: bool = True,
    j3: bool = False,
    dt_max_s: float = 60.0,
    first_pass_gap_s: float = 6 * 3600.0,
) -> dict:
    """
    9-state LKF with Dynamic Model Compensation (DMC).

    Internal error-state:
      dx = [dr(3), dv(3), dw(3)]

    Reference-state:
      X* = [r(3), v(3), w(3)]

    Returns a dict aligned with the existing 6-state post-processing keys while
    also including 9-state histories for DMC-specific analysis.
    """
    X0_star = np.asarray(X0_star, dtype=float).reshape(6,)
    P0 = np.asarray(P0, dtype=float).reshape(9, 9)
    R = np.asarray(R, dtype=float).reshape(2, 2)

    tau_s = float(tau_s)
    if tau_s <= 0.0:
        raise ValueError(f"tau_s must be > 0, got {tau_s}")
    dt_max_s = float(dt_max_s)
    if dt_max_s <= 0.0:
        raise ValueError(f"dt_max_s must be > 0, got {dt_max_s}")
    sigma_m_s2 = float(sigma_accel_m_s2)
    if sigma_m_s2 < 0.0:
        raise ValueError(f"sigma_accel_m_s2 must be >= 0, got {sigma_m_s2}")

    beta = 1.0 / tau_s
    B = beta * np.eye(3, dtype=float)

    sigma_km_s2 = sigma_m_s2 * 1.0e-3
    q_drive = 2.0 * beta * (sigma_km_s2**2)
    Qc = q_drive * np.eye(3, dtype=float)

    L = np.zeros((9, 3), dtype=float)
    L[6:9, 0:3] = np.eye(3, dtype=float)

    X0_ref = np.hstack((X0_star, np.zeros(3, dtype=float)))

    station_map = {st.name: st for st in stations}
    all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
    N = len(all_meas)
    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
    st_meas = [m["station"] for m in all_meas]

    if Xtrue_meas is not None:
        Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
        if Xtrue_meas.shape != (N, 6):
            raise ValueError(f"Xtrue_meas must have shape ({N}, 6), got {Xtrue_meas.shape}")

    settings = PropSettings(rtol=float(reltol), atol=float(abstol), method=str(method))

    f = lambda t, x: _dmc_ref_dynamics(
        t,
        x,
        mu=float(mu),
        J2=float(J2),
        J3=float(J3),
        B=B,
        Re=float(Re),
        j2=bool(j2),
        j3=bool(j3),
    )
    Xstar_hist, Xstar_dense = _propagate_reference_history(
        x0=X0_ref,
        t_eval=t_meas,
        rhs=f,
        settings=settings,
    )
    if Xstar_dense is None:
        raise RuntimeError("DMC reference propagation did not return dense output.")

    prefit = np.full((N, 2), np.nan, dtype=float)
    postfit_lin = np.full((N, 2), np.nan, dtype=float)
    postfit_nl = np.full((N, 2), np.nan, dtype=float)

    X_pf9 = np.full((N, 9), np.nan, dtype=float)
    P_meas9 = np.full((N, 9, 9), np.nan, dtype=float)
    two_sigma9 = np.full((N, 9), np.nan, dtype=float)

    X_pf6 = np.full((N, 6), np.nan, dtype=float)
    P_meas6 = np.full((N, 6, 6), np.nan, dtype=float)
    P_pf6 = np.full((N, 36), np.nan, dtype=float)
    two_sigma6 = np.full((N, 6), np.nan, dtype=float)
    Qk_interval9 = np.full((N, 9, 9), np.nan, dtype=float)
    Qk_interval_ww_diag = np.full((N, 3), np.nan, dtype=float)

    state_error6 = None if Xtrue_meas is None else np.full((N, 6), np.nan, dtype=float)

    x_hat = np.zeros(9, dtype=float)
    P = P0.copy()
    num_updates = 0
    num_skips = 0

    for j in range(N):
        t = float(t_meas[j])
        st = station_map[st_meas[j]]
        Y = np.array([all_meas[j]["rho_km"], all_meas[j]["rho_dot_km_s"]], dtype=float)

        Xstar = Xstar_hist[j, :]

        xbar = x_hat.copy()
        Pbar = P.copy()
        Qeq_interval = np.zeros((9, 9), dtype=float)
        if j > 0:
            t_prev = float(t_meas[j - 1])
            t_internal = t_prev
            while t_internal < (t - 1.0e-12):
                h = min(dt_max_s, t - t_internal)
                Xstar_lin = np.asarray(Xstar_dense(t_internal), dtype=float).reshape(9,)
                Aref = _dmc_A_matrix(
                    Xstar_lin,
                    mu=float(mu),
                    J2=float(J2),
                    J3=float(J3),
                    B=B,
                    Re=float(Re),
                    j2=bool(j2),
                    j3=bool(j3),
                )
                Phi, Qk = _van_loan_discretization(Aref, L, Qc, h)
                xbar = Phi @ xbar
                Pbar = Phi @ Pbar @ Phi.T + Qk
                Pbar = 0.5 * (Pbar + Pbar.T)
                Qeq_interval = Phi @ Qeq_interval @ Phi.T + Qk
                Qeq_interval = 0.5 * (Qeq_interval + Qeq_interval.T)
                if not np.all(np.isfinite(Pbar)):
                    raise RuntimeError(
                        f"DMC covariance became non-finite during propagation at measurement index {j}, "
                        f"t={t:.3f} s, substep start={t_internal:.3f} s"
                    )
                t_internal += h

        C = _measurement_model(st, Xstar[:6], t)
        if C is not None:
            OminusC = Y - C
            prefit[j, :] = OminusC

            Rs, Vs, _ = st.ecef2eci(t, st.r_ecef, np.zeros(3))
            H6 = H_range_rangerate(Xstar[:3], Xstar[3:6], Rs, Vs)
            H9 = np.hstack((H6, np.zeros((2, 3), dtype=float)))

            S = H9 @ Pbar @ H9.T + R
            if not np.all(np.isfinite(S)):
                raise RuntimeError(f"DMC innovation covariance became non-finite at measurement index {j}, t={t:.3f} s")
            try:
                Sinv = np.linalg.solve(S, np.eye(2))
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"DMC innovation covariance solve failed at measurement index {j}, t={t:.3f} s: {exc}"
                ) from exc
            K = Pbar @ H9.T @ Sinv

            x_hat = xbar + K @ (OminusC - H9 @ xbar)
            IKH = np.eye(9) - K @ H9
            P = IKH @ Pbar @ IKH.T + K @ R @ K.T
            P = 0.5 * (P + P.T)
            if not np.all(np.isfinite(P)):
                raise RuntimeError(f"DMC updated covariance became non-finite at measurement index {j}, t={t:.3f} s")
            num_updates += 1

            postfit_lin[j, :] = OminusC - (H9 @ x_hat)
        else:
            x_hat = xbar
            P = Pbar
            num_skips += 1

        Xpost = Xstar + x_hat
        X_pf9[j, :] = Xpost
        X_pf6[j, :] = Xpost[:6]
        Qk_interval9[j, :, :] = Qeq_interval
        Qk_interval_ww_diag[j, :] = np.diag(Qeq_interval[6:9, 6:9])

        P_meas9[j, :, :] = P
        P_meas6[j, :, :] = P[:6, :6]
        P_pf6[j, :] = P[:6, :6].reshape(-1, order="F")

        two_sigma9[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P), 0.0))
        two_sigma6[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P[:6, :6]), 0.0))

        C_post = _measurement_model(st, Xpost[:6], t)
        if C_post is not None:
            postfit_nl[j, :] = Y - C_post

        if state_error6 is not None:
            state_error6[j, :] = Xpost[:6] - Xtrue_meas[j, :]

    rms_final = KalmanFilterBase.compute_rms_summary_from_arrays(
        t=t_meas,
        postfit=postfit_nl,
        state_err=state_error6,
        first_pass_gap_s=float(first_pass_gap_s),
    )

    return {
        "t_meas": t_meas,
        "station_meas": st_meas,
        "xhat_meas": X_pf6,
        "Xhat_meas": X_pf6,
        "X_pf": X_pf6,
        "state_error_meas": state_error6,
        "prefit_resids_final": prefit,
        "postfit_resids_linear_final": postfit_lin,
        "postfit_resids_meas": postfit_nl,
        "P_meas": P_meas6,
        "Phat_meas": P_meas6,
        "P_pf": P_pf6,
        "two_sigma_meas": two_sigma6,
        "rms_final": rms_final,
        "rms_by_iter": None,
        "Xhat_meas_9": X_pf9,
        "P_meas_9": P_meas9,
        "two_sigma_meas_9": two_sigma9,
        "Qk_interval_9": Qk_interval9,
        "Qk_interval_ww_diag": Qk_interval_ww_diag,
        "dmc_tau_s": tau_s,
        "dmc_beta_1_s": beta,
        "dmc_sigma_m_s2": sigma_m_s2,
        "dmc_q_psd_km2_s5": q_drive,
        "dmc_num_updates": num_updates,
        "dmc_num_skipped": num_skips,
    }


def ekf_with_dmc(
    *,
    all_meas,
    stations,
    X0_hat: np.ndarray,
    P0: np.ndarray,
    R: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    tau_s: float,
    sigma_accel_m_s2: float,
    Xtrue_meas: np.ndarray | None = None,
    Re: float = 6378.0,
    reltol: float = 1.0e-10,
    abstol: float = 1.0e-10,
    method: str = "DOP853",
    j2: bool = True,
    j3: bool = False,
    dt_max_s: float = 60.0,
    first_pass_gap_s: float = 6 * 3600.0,
    bootstrap_steps: int = 0,
) -> dict:
    """
    9-state EKF with Dynamic Model Compensation (DMC).

    Estimated state:
      X = [r(3), v(3), w(3)]
    """
    X0_hat = np.asarray(X0_hat, dtype=float).reshape(6,)
    P0 = np.asarray(P0, dtype=float).reshape(9, 9)
    R = np.asarray(R, dtype=float).reshape(2, 2)

    tau_s = float(tau_s)
    if tau_s <= 0.0:
        raise ValueError(f"tau_s must be > 0, got {tau_s}")
    dt_max_s = float(dt_max_s)
    if dt_max_s <= 0.0:
        raise ValueError(f"dt_max_s must be > 0, got {dt_max_s}")
    sigma_m_s2 = float(sigma_accel_m_s2)
    if sigma_m_s2 < 0.0:
        raise ValueError(f"sigma_accel_m_s2 must be >= 0, got {sigma_m_s2}")
    bootstrap_steps = int(bootstrap_steps)

    beta = 1.0 / tau_s
    B = beta * np.eye(3, dtype=float)

    # sigma interpreted as steady-state std-dev of uncompensated acceleration.
    sigma_km_s2 = sigma_m_s2 * 1.0e-3
    q_drive = 2.0 * beta * (sigma_km_s2**2)
    Qc = q_drive * np.eye(3, dtype=float)

    L = np.zeros((9, 3), dtype=float)
    L[6:9, 0:3] = np.eye(3, dtype=float)

    station_map = {st.name: st for st in stations}
    all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
    N = len(all_meas)
    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
    st_meas = [m["station"] for m in all_meas]

    if Xtrue_meas is not None:
        Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
        if Xtrue_meas.shape != (N, 6):
            raise ValueError(f"Xtrue_meas must have shape ({N}, 6), got {Xtrue_meas.shape}")

    settings = PropSettings(rtol=float(reltol), atol=float(abstol), method=str(method))

    f = lambda t, x: _dmc_ref_dynamics(
        t,
        x,
        mu=float(mu),
        J2=float(J2),
        J3=float(J3),
        B=B,
        Re=float(Re),
        j2=bool(j2),
        j3=bool(j3),
    )

    prefit = np.full((N, 2), np.nan, dtype=float)
    postfit_lin = np.full((N, 2), np.nan, dtype=float)
    postfit_nl = np.full((N, 2), np.nan, dtype=float)

    X_pf9 = np.full((N, 9), np.nan, dtype=float)
    P_meas9 = np.full((N, 9, 9), np.nan, dtype=float)
    two_sigma9 = np.full((N, 9), np.nan, dtype=float)
    Qk_interval9 = np.full((N, 9, 9), np.nan, dtype=float)
    Qk_interval_ww_diag = np.full((N, 3), np.nan, dtype=float)

    X_pf6 = np.full((N, 6), np.nan, dtype=float)
    P_meas6 = np.full((N, 6, 6), np.nan, dtype=float)
    P_pf6 = np.full((N, 36), np.nan, dtype=float)
    two_sigma6 = np.full((N, 6), np.nan, dtype=float)

    state_error6 = None if Xtrue_meas is None else np.full((N, 6), np.nan, dtype=float)

    X_hat = np.hstack((X0_hat, np.zeros(3, dtype=float)))
    P = P0.copy()
    num_updates = 0
    num_skips = 0
    t_prev = float(t_meas[0]) if N > 0 else 0.0

    for j in range(N):
        t = float(t_meas[j])
        st = station_map[st_meas[j]]
        Y = np.array([all_meas[j]["rho_km"], all_meas[j]["rho_dot_km_s"]], dtype=float)

        Xbar = X_hat.copy()
        Pbar = P.copy()
        Qeq_interval = np.zeros((9, 9), dtype=float)
        if j > 0:
            t_internal = t_prev
            while t_internal < (t - 1.0e-12):
                h = min(dt_max_s, t - t_internal)
                X_start = Xbar.copy()
                Xbar = _propagate_state_step(
                    x0=Xbar,
                    t0=t_internal,
                    t1=t_internal + h,
                    rhs=f,
                    settings=settings,
                )
                Aref = _dmc_A_matrix(
                    X_start,
                    mu=float(mu),
                    J2=float(J2),
                    J3=float(J3),
                    B=B,
                    Re=float(Re),
                    j2=bool(j2),
                    j3=bool(j3),
                )
                Phi, Qk = _van_loan_discretization(Aref, L, Qc, h)
                Pbar = Phi @ Pbar @ Phi.T + Qk
                Pbar = 0.5 * (Pbar + Pbar.T)
                Qeq_interval = Phi @ Qeq_interval @ Phi.T + Qk
                Qeq_interval = 0.5 * (Qeq_interval + Qeq_interval.T)
                if not np.all(np.isfinite(Pbar)):
                    raise RuntimeError(
                        f"DMC EKF covariance became non-finite during propagation at measurement index {j}, "
                        f"t={t:.3f} s, substep start={t_internal:.3f} s"
                    )
                t_internal += h

        C = _measurement_model(st, Xbar[:6], t)
        if C is not None:
            OminusC = Y - C
            prefit[j, :] = OminusC

            Rs, Vs, _ = st.ecef2eci(t, st.r_ecef, np.zeros(3))
            H6 = H_range_rangerate(Xbar[:3], Xbar[3:6], Rs, Vs)
            H9 = np.hstack((H6, np.zeros((2, 3), dtype=float)))

            S = H9 @ Pbar @ H9.T + R
            if not np.all(np.isfinite(S)):
                raise RuntimeError(f"DMC EKF innovation covariance became non-finite at measurement index {j}, t={t:.3f} s")
            try:
                Sinv = np.linalg.solve(S, np.eye(2))
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"DMC EKF innovation covariance solve failed at measurement index {j}, t={t:.3f} s: {exc}"
                ) from exc
            K = Pbar @ H9.T @ Sinv

            dX = K @ OminusC
            X_hat = Xbar + dX
            IKH = np.eye(9) - K @ H9
            P = IKH @ Pbar @ IKH.T + K @ R @ K.T
            P = 0.5 * (P + P.T)
            if not np.all(np.isfinite(P)):
                raise RuntimeError(f"DMC EKF updated covariance became non-finite at measurement index {j}, t={t:.3f} s")
            num_updates += 1

            postfit_lin[j, :] = OminusC - (H9 @ dX)
        else:
            X_hat = Xbar
            P = Pbar
            num_skips += 1

        # Optional rectification behavior, similar to friend's bootstrap idea.
        if bootstrap_steps > 0 and j < bootstrap_steps:
            X_hat[6:9] = 0.0

        X_pf9[j, :] = X_hat
        X_pf6[j, :] = X_hat[:6]
        Qk_interval9[j, :, :] = Qeq_interval
        Qk_interval_ww_diag[j, :] = np.diag(Qeq_interval[6:9, 6:9])

        P_meas9[j, :, :] = P
        P_meas6[j, :, :] = P[:6, :6]
        P_pf6[j, :] = P[:6, :6].reshape(-1, order="F")

        two_sigma9[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P), 0.0))
        two_sigma6[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P[:6, :6]), 0.0))

        C_post = _measurement_model(st, X_hat[:6], t)
        if C_post is not None:
            postfit_nl[j, :] = Y - C_post

        if state_error6 is not None:
            state_error6[j, :] = X_hat[:6] - Xtrue_meas[j, :]

        t_prev = t

    rms_final = KalmanFilterBase.compute_rms_summary_from_arrays(
        t=t_meas,
        postfit=postfit_nl,
        state_err=state_error6,
        first_pass_gap_s=float(first_pass_gap_s),
    )

    return {
        "t_meas": t_meas,
        "station_meas": st_meas,
        "xhat_meas": X_pf6,
        "Xhat_meas": X_pf6,
        "X_pf": X_pf6,
        "state_error_meas": state_error6,
        "prefit_resids_final": prefit,
        "postfit_resids_linear_final": postfit_lin,
        "postfit_resids_meas": postfit_nl,
        "P_meas": P_meas6,
        "Phat_meas": P_meas6,
        "P_pf": P_pf6,
        "two_sigma_meas": two_sigma6,
        "rms_final": rms_final,
        "rms_by_iter": None,
        "Xhat_meas_9": X_pf9,
        "P_meas_9": P_meas9,
        "two_sigma_meas_9": two_sigma9,
        "Qk_interval_9": Qk_interval9,
        "Qk_interval_ww_diag": Qk_interval_ww_diag,
        "dmc_tau_s": tau_s,
        "dmc_beta_1_s": beta,
        "dmc_sigma_m_s2": sigma_m_s2,
        "dmc_q_psd_km2_s5": q_drive,
        "dmc_num_updates": num_updates,
        "dmc_num_skipped": num_skips,
    }
