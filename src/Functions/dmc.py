from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .jacobians import accel_wJ2J3, dadr_wJ2J3
from .propagation import PropSettings, phi_i0_to_phi_steps, propagate_x_phi_history
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
    A = lambda t, x: _dmc_A_matrix(
        x,
        mu=float(mu),
        J2=float(J2),
        J3=float(J3),
        B=B,
        Re=float(Re),
        j2=bool(j2),
        j3=bool(j3),
    )

    Xstar_hist, Phi_i0_hist = propagate_x_phi_history(
        x0=X0_ref,
        t_eval=t_meas,
        f=f,
        A=A,
        settings=settings,
    )
    Phi_step = phi_i0_to_phi_steps(Phi_i0_hist)

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

    state_error6 = None if Xtrue_meas is None else np.full((N, 6), np.nan, dtype=float)

    x_hat = np.zeros(9, dtype=float)
    P = P0.copy()

    for j in range(N):
        t = float(t_meas[j])
        st = station_map[st_meas[j]]
        Y = np.array([all_meas[j]["rho_km"], all_meas[j]["rho_dot_km_s"]], dtype=float)

        Xstar = Xstar_hist[j, :]
        Phi = Phi_step[j, :, :]

        dt = 0.0 if j == 0 else (t - float(t_meas[j - 1]))
        Aref = _dmc_A_matrix(
            Xstar,
            mu=float(mu),
            J2=float(J2),
            J3=float(J3),
            B=B,
            Re=float(Re),
            j2=bool(j2),
            j3=bool(j3),
        )
        _, Qk = _van_loan_discretization(Aref, L, Qc, dt)

        xbar = Phi @ x_hat
        Pbar = Phi @ P @ Phi.T + Qk

        C = _measurement_model(st, Xstar[:6], t)
        if C is not None:
            OminusC = Y - C
            prefit[j, :] = OminusC

            Rs, Vs, _ = st.ecef2eci(t, st.r_ecef, np.zeros(3))
            H6 = H_range_rangerate(Xstar[:3], Xstar[3:6], Rs, Vs)
            H9 = np.hstack((H6, np.zeros((2, 3), dtype=float)))

            S = H9 @ Pbar @ H9.T + R
            K = Pbar @ H9.T @ np.linalg.solve(S, np.eye(2))

            x_hat = xbar + K @ (OminusC - H9 @ xbar)
            IKH = np.eye(9) - K @ H9
            P = IKH @ Pbar @ IKH.T + K @ R @ K.T

            postfit_lin[j, :] = OminusC - (H9 @ x_hat)
        else:
            x_hat = xbar
            P = Pbar

        Xpost = Xstar + x_hat
        X_pf9[j, :] = Xpost
        X_pf6[j, :] = Xpost[:6]

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
        "dmc_tau_s": tau_s,
        "dmc_beta_1_s": beta,
        "dmc_sigma_m_s2": sigma_m_s2,
        "dmc_q_psd_km2_s5": q_drive,
    }
