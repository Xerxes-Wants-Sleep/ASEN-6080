from __future__ import annotations

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp

from .filters import KalmanFilterBase
from .jacobians import stm
from .range_rangerate import H_range_rangerate


def _qr_householder_transform(A: np.ndarray, fix_sign: bool = True) -> np.ndarray:
    """
    SRIF orthogonal transform:
      - Triangularize only the first (p-1) columns (the 'design' block)
      - Apply the same Q^T to the entire augmented matrix so the last column
        becomes [b; e] with e being the post-fit residual(s).
    """
    A = np.asarray(A, dtype=float)
    m, p = A.shape
    ntri = p - 1  # number of columns to triangularize (all but RHS)

    # QR of left block only
    Q, _ = la.qr(A[:, :ntri], mode="full", check_finite=False)

    # Apply Q^T to the full augmented matrix
    out = Q.T @ A

    # Optional: enforce positive diagonal on the triangular block
    if fix_sign:
        d = np.diag(out[:ntri, :ntri])
        for i in range(ntri):
            if d[i] < 0:
                out[i, :] *= -1.0

    # Optional: hard-zero tiny subdiagonal entries in the triangular block
    out[:ntri, :ntri] = np.triu(out[:ntri, :ntri])

    return out


def _info_factor_from_cov(P: np.ndarray) -> np.ndarray:
    """
    Given covariance P, return upper-triangular R such that:
        R^T R = P^{-1}
    Do this without forming P^{-1} explicitly:
        P = U^T U (U upper) -> P^{-1} = U^{-1} U^{-T}
        so R = U^{-1}.
    """
    P = np.asarray(P, dtype=float)
    U = la.cholesky(P, lower=False, check_finite=False)  # P = U^T U
    I = np.eye(P.shape[0])
    R = la.solve_triangular(U, I, lower=False, check_finite=False)  # R = U^{-1}
    return R


def _cov_from_info_factor(R: np.ndarray) -> np.ndarray:
    """
    Given upper-triangular R with R^T R = P^{-1}, recover P:
        P = (R^T R)^{-1} = R^{-1} R^{-T}
    """
    R = np.asarray(R, dtype=float)
    n = R.shape[0]
    I = np.eye(n)
    Rinv = la.solve_triangular(R, I, lower=False, check_finite=False)
    return Rinv @ Rinv.T


class SquareRootInformationFilter(KalmanFilterBase):
    """
    SRIF in error-state form about a precomputed reference trajectory X*(t).

    Matches your LKF structure:
      1) propagate X*(t_i), Phi(t_i,t0) ONCE
      2) form step STM Phi(t_i,t_{i-1}) by matrix solve
      3) run sequential filter

    SRIF differences:
      - store/update R (upper) and b where R^T R = P^{-1}, b = R x
      - time update uses Rbar = R * Phi^{-1}, bbar = b
      - measurement update stacks [R b; Htilde y] then QR -> new R,b and postfit e

    Optional process noise:
      - If you have a discrete Q_k (nxn), we can treat it as additive noise on x:
            x_k = Phi x_{k-1} + u_k,  u_k~N(uBar, Q_k)
        then use Gamma=I, q=n in the SRIF time-update stack.
      - If you have an accel-noise model (q=3) and want Gamma(dt)=[dt/2 I; I],
        you can pass Q_acc_3x3 and uBar_3, and set process_noise_mode="accel".
    """

    def __init__(
        self,
        X0_star: np.ndarray,
        P0: np.ndarray,
        R: np.ndarray,
        Q: np.ndarray,
        mu: float,
        J2: float,
        J3: float,
        Re: float = 6378.0,
        reltol: float = 1e-10,
        abstol: float = 1e-10,
        method: str = "DOP853",
        j2: bool = True,
        j3: bool = False,
        first_pass_gap_s: float = 6 * 3600.0,
        recenter_reference: bool = False,
        
        # SRIF Specific Options:
        process_noise_mode: str = "none",  # "none" | "additive" | "accel"
        uBar: np.ndarray | None = None,    # mean noise (q,)
        max_process_dt_s: float = 10.0,    # mimic your class code gating if you want
    ):
        super().__init__(X0_star, P0, R, Q)

        self.X0_star = np.asarray(X0_star, dtype=float).reshape(6,).copy()
        self.mu = float(mu)
        self.J2 = float(J2)
        self.J3 = float(J3)
        self.Re = float(Re)
        self.reltol = float(reltol)
        self.abstol = float(abstol)
        self.method = str(method)
        self.j2 = bool(j2)
        self.j3 = bool(j3)
        self.first_pass_gap_s = float(first_pass_gap_s)
        self.recenter_reference = bool(recenter_reference)

        self.process_noise_mode = str(process_noise_mode).lower()
        self.uBar = None if uBar is None else np.asarray(uBar, dtype=float).reshape(-1)
        self.max_process_dt_s = float(max_process_dt_s)

    # ---------- Reference + STM propagation ----------
    def propagate_state_and_stm_history(self, t_eval: np.ndarray):
        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
        t0 = float(t_eval[0])
        tf = float(t_eval[-1])

        nx = 6
        remove = np.array([6, 7, 8], dtype=int)

        X0_9 = np.hstack((self.X0_star, self.mu, self.J2, self.J3))
        Phi0 = np.eye(nx)
        y0 = np.hstack((X0_9, Phi0.reshape(-1)))

        def fun(t, y):
            return stm(
                t,
                state9=y[:9],
                phi=y[9:].reshape(nx, nx),
                rows_col_to_remove=remove,
                Re=self.Re,
                j2=self.j2,
                j3=self.j3,
            )

        sol = solve_ivp(
            fun,
            (t0, tf),
            y0,
            t_eval=t_eval,
            rtol=self.reltol,
            atol=self.abstol,
            method=self.method,
        )
        if not sol.success:
            raise RuntimeError(f"STM integration failed: {sol.message}")

        Y = sol.y.T
        Xstar_hist = Y[:, :6]
        Phi_i0_hist = Y[:, 9:].reshape(-1, nx, nx)  # Phi(t_i, t0)
        return Xstar_hist, Phi_i0_hist

    @staticmethod
    def phi_i0_to_phi_step(Phi_i0_hist: np.ndarray):
        Phi_i0_hist = np.asarray(Phi_i0_hist, dtype=float)
        N = Phi_i0_hist.shape[0]
        Phi_step = np.zeros_like(Phi_i0_hist)
        Phi_step[0] = np.eye(6)

        I = np.eye(6)
        for i in range(1, N):
            Phi_step[i] = Phi_i0_hist[i] @ np.linalg.solve(Phi_i0_hist[i - 1], I)

        return Phi_step

    @staticmethod
    def G(station, X6: np.ndarray, t: float):
        d = station.measure(X6[:3], X6[3:], float(t))
        if d is None:
            return None
        rho = d["rho_km"] if "rho_km" in d else d["rho"]
        rhod = d["rho_dot_km_s"] if "rho_dot_km_s" in d else d["rho_dot"]
        return np.array([rho, rhod], dtype=float)

    @staticmethod
    def _gamma_accel(dt: float) -> np.ndarray:
        """Gamma(dt) for integrated accel noise into [r; v] 6-state."""
        dt = float(dt)
        return np.vstack([(0.5 * dt * dt) * np.eye(3), dt * np.eye(3)])  # (6,3)

    # ---------- SRIF run ----------
    def run(self, all_meas, stations, Xtrue_meas: np.ndarray | None = None):
        station_map = {st.name: st for st in stations}
        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
        t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
        st_meas = [m["station"] for m in all_meas]
        N = len(all_meas)

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (N, 6):
                raise ValueError(f"Xtrue_meas must have shape ({N}, 6) aligned with sorted measurement order.")

        # 1) Precompute reference + Phi steps (same as your LKF)
        Xstar_hist, Phi_i0_hist = self.propagate_state_and_stm_history(t_meas)
        Phi_step = self.phi_i0_to_phi_step(Phi_i0_hist)

        # 2) Measurement whitening factor (constant R assumed)
        V = la.cholesky(self.R, lower=True, check_finite=False)  # R = V V^T

        # 3) Allocate outputs (keep names similar to your LKF dict)
        residuals = np.full((N, 2), np.nan, dtype=float)          # prefit (unwhitened): O - C_ref
        postfit_lin = np.full((N, 2), np.nan, dtype=float)        # linear postfit (unwhitened): V*e
        postfit_nl = np.full((N, 2), np.nan, dtype=float)         # placeholder for plotting compatibility
        X_pf = np.full((N, 6), np.nan, dtype=float)
        P_meas = np.full((N, 6, 6), np.nan, dtype=float)
        P_pf = np.full((N, 36), np.nan, dtype=float)
        two_sigma = np.full((N, 6), np.nan, dtype=float)
        state_error = None if Xtrue_meas is None else np.full((N, 6), np.nan, dtype=float)

        prefit_whitened = np.full((N, 2), np.nan, dtype=float)
        postfit_whitened = np.full((N, 2), np.nan, dtype=float)

        P_pred_hist = np.full((N, 6, 6), np.nan, dtype=float)
        x_pred_hist = np.full((N, 6), np.nan, dtype=float)

        # SRIF smoothing bookkeeping (optional)
        Ru_hist, Rux_hist, bTildeu_hist, uHat_hist = [], [], [], []

        # 4) Initialize SRIF prior: R0^T R0 = P0^{-1}, b0 = R0 x0
        n = 6
        x_hat = np.zeros(6, dtype=float)  # error-state estimate (same as LKF initial)
        R_info = _info_factor_from_cov(self.P0)  # (6,6) upper
        b_info = R_info @ x_hat                 # (6,)

        # Process noise state for SRIF time-update stack
        bu_prev = None
        if self.process_noise_mode in ("additive", "accel"):
            if self.uBar is None:
                raise ValueError("SRIF: process_noise_mode != 'none' requires uBar (mean noise vector).")

        # Loop
        for k in range(N):
            t = float(t_meas[k])
            st = station_map[st_meas[k]]
            Xstar = Xstar_hist[k, :]
            Phi = Phi_step[k, :, :]

            # ---- Time update in SRIF ----
            if k == 0:
                dt = 0.0
            else:
                dt = t - float(t_meas[k - 1])

            # Compute Rtilde = R_{k-1} * Phi^{-1} without explicit inverse:
            # Rtilde = (Phi^{-T} R^T)^T
            Rtilde = la.solve(Phi.T, R_info.T, check_finite=False).T
            btilde = b_info.copy()

            # Optional SRIF process noise time update
            use_pn = (self.process_noise_mode != "none") and (dt > 0.0) and (dt <= self.max_process_dt_s)

            if use_pn:
                if self.process_noise_mode == "additive":
                    # Use discrete Qk (6x6) as additive noise covariance on x.
                    # Prefer your existing build_process_noise if available.
                    if hasattr(self, "build_process_noise"):
                        Qk = self.build_process_noise(dt, r_eci=Xstar[:3], v_eci=Xstar[3:6])
                    else:
                        Qk = np.array(self.Q, dtype=float).copy()

                    Qk = np.asarray(Qk, dtype=float)
                    if Qk.shape != (n, n):
                        raise ValueError(f"SRIF additive noise expects Qk shape {(n,n)}, got {Qk.shape}")

                    q = n
                    Gamma = np.eye(n)
                    Ru = _info_factor_from_cov(Qk)  # Ru^T Ru = Qk^{-1}
                    if bu_prev is None:
                        bu_prev = Ru @ self.uBar  # initial
                elif self.process_noise_mode == "accel":
                    # Accel noise model: q=3, Gamma(dt)=(6x3)
                    Qacc = np.asarray(self.Q, dtype=float)
                    if Qacc.shape != (3, 3):
                        raise ValueError(f"SRIF accel noise expects Q as (3,3), got {Qacc.shape}")
                    q = 3
                    Gamma = self._gamma_accel(dt)  # (6,3)
                    Ru = _info_factor_from_cov(Qacc)
                    if bu_prev is None:
                        bu_prev = Ru @ self.uBar
                else:
                    raise ValueError("process_noise_mode must be 'none', 'additive', or 'accel'.")

                # Stack (matches your MATLAB structure):
                # [ Ru     0      bu_prev ]
                # [ -Rtilde*Gamma  Rtilde  b_prev ]
                top = np.hstack([Ru, np.zeros((q, n)), bu_prev.reshape(-1, 1)])
                bot = np.hstack([-Rtilde @ Gamma, Rtilde, b_info.reshape(-1, 1)])
                M = np.vstack([top, bot])

                out = _qr_householder_transform(M, fix_sign=True)

                # Extract updated blocks
                Ru_k = out[:q, :q]
                Rux_k = out[:q, q:q+n]
                bTildeu_k = out[:q, q+n]

                Rtilde = out[q:q+n, q:q+n]
                btilde = out[q:q+n, q+n]

                # Update bu_prev for next epoch
                bu_prev = Ru_k @ self.uBar

                Ru_hist.append(Ru_k)
                Rux_hist.append(Rux_k)
                bTildeu_hist.append(bTildeu_k)
            else:
                # Force upper-triangular (optional but recommended)
                M = np.hstack([Rtilde, btilde.reshape(-1, 1)])
                out = _qr_householder_transform(M, fix_sign=True)
                Rtilde = out[:n, :n]
                btilde = out[:n, n]

                Ru_hist.append(None)
                Rux_hist.append(None)
                bTildeu_hist.append(None)

            # Predicted covariance (for stats/plots, not needed for SRIF core)
            P_pred = _cov_from_info_factor(Rtilde)
            P_pred_hist[k, :, :] = P_pred
            x_pred = la.solve_triangular(Rtilde, btilde, lower=False, check_finite=False)
            x_pred_hist[k, :] = x_pred

            # ---- Measurement update ----
            Y = np.array([all_meas[k]["rho_km"], all_meas[k]["rho_dot_km_s"]], dtype=float)

            C_ref = self.G(st, Xstar, t)
            if C_ref is None:
                # No measurement update: carry time-update results
                x_hat = x_pred
                R_info = Rtilde
                b_info = btilde
                X_post = Xstar + x_hat

                X_pf[k, :] = X_post
                P_post = _cov_from_info_factor(R_info)
                P_meas[k, :, :] = P_post
                P_pf[k, :] = P_post.reshape(-1, order="F")
                two_sigma[k, :] = 2.0 * np.sqrt(np.maximum(np.diag(P_post), 0.0))

                if state_error is not None:
                    state_error[k, :] = X_post - Xtrue_meas[k, :]

                uHat_hist.append(None)
                self.Xhat = X_post
                self.Phat = P_post
                self.log_epoch(
                    t,
                    postfit_resid=np.array([np.nan, np.nan], dtype=float),
                    Xtrue=(None if Xtrue_meas is None else Xtrue_meas[k, :]),
                )
                continue

            # Prefit residual (unwhitened)
            OminusC = Y - C_ref
            residuals[k, :] = OminusC

            # Whiten residual and H
            y = la.solve_triangular(V, OminusC, lower=True, check_finite=False)  # (2,)
            prefit_whitened[k, :] = y

            Rs, Vs, _ = st.ecef2eci(t, st.r_ecef, np.zeros(3))
            H = H_range_rangerate(Xstar[:3], Xstar[3:], Rs, Vs)  # (2,6)

            Htilde = la.solve_triangular(V, H, lower=True, check_finite=False)  # (2,6)

            # Stack and QR (Householder) measurement update:
            # [ Rtilde  btilde ]
            # [ Htilde  y      ]
            M = np.vstack([
                np.hstack([Rtilde, btilde.reshape(-1, 1)]),
                np.hstack([Htilde, y.reshape(-1, 1)]),
            ])

            out = _qr_householder_transform(M, fix_sign=True)

            R_post = out[:n, :n]
            b_post = out[:n, n]
            e = out[n:, n]  # whitened post-fit residual (size 2)
            postfit_whitened[k, :] = e

            # Solve x_hat = R^{-1} b
            x_hat = la.solve_triangular(R_post, b_post, lower=False, check_finite=False)

            # Optional retrospective process noise estimate uHat_{k-1}
            if use_pn and (Ru_hist[-1] is not None):
                Ru_k = Ru_hist[-1]
                Rux_k = Rux_hist[-1]
                bTildeu_k = bTildeu_hist[-1]
                u_hat = la.solve_triangular(Ru_k, (bTildeu_k - Rux_k @ x_hat), lower=False, check_finite=False)
                uHat_hist.append(u_hat)
            else:
                uHat_hist.append(None)

            # Save
            R_info = R_post
            b_info = b_post

            X_post = Xstar + x_hat
            X_pf[k, :] = X_post

            P_post = _cov_from_info_factor(R_info)
            P_meas[k, :, :] = P_post
            P_pf[k, :] = P_post.reshape(-1, order="F")
            two_sigma[k, :] = 2.0 * np.sqrt(np.maximum(np.diag(P_post), 0.0))

            postfit_lin[k, :] = V @ e  # unwhitened postfit residual

            if state_error is not None:
                state_error[k, :] = X_post - Xtrue_meas[k, :]

            # Keep base-class history consistent with your RMS summary
            self.Xhat = X_post
            self.Phat = P_post
            self.log_epoch(t, postfit_resid=postfit_lin[k, :],
                           Xtrue=(None if Xtrue_meas is None else Xtrue_meas[k, :]))

        rms_final = self.print_rms_summary(label="SRIF", first_pass_gap_s=self.first_pass_gap_s)

        return {
            "t_meas": t_meas,
            "station_meas": st_meas,

            # states
            "xhat_meas": X_pf,
            "Xhat_meas": X_pf,
            "X_pf": X_pf,

            # residuals
            "prefit_resids_final": residuals,            # unwhitened O-C
            "postfit_resids_linear_final": postfit_lin,  # unwhitened V*e
            "postfit_resids_meas": postfit_nl,
            "prefit_res_whitened": prefit_whitened,
            "postfit_res_whitened": postfit_whitened,

            # covariance
            "P_meas": P_meas,
            "Phat_meas": P_meas,
            "P_pf": P_pf,
            "P_pred_hist": P_pred_hist,
            "x_pred_hist": x_pred_hist,
            "Phi_step": Phi_step,

            "two_sigma_meas": two_sigma,
            "state_error_meas": state_error,

            # SRIF process-noise bookkeeping (only meaningful if process_noise_mode not none)
            "Ru": Ru_hist,
            "Rux": Rux_hist,
            "bTildeu": bTildeu_hist,
            "uHat": uHat_hist,

            "rms_final": rms_final,
            "rms_by_iter": None,
        }
