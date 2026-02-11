from __future__ import annotations

import numpy as np

from .filter_18_state import KalmanFilterBase
from .propagation import propagate_x_phi_history, PropSettings
from .range_rangerate import H_range_rangerate, H_tilde_range_rangerate_augmented


class PotterLKF18State(KalmanFilterBase):
    """
    Linearized Kalman Filter (18-state) with Potter square-root updates.

    Error-state filter about a precomputed reference trajectory X*(t):
      - Precompute X*(t_i) and Phi(t_i,t0)
      - Use step STM Phi(t_i,t_{i-1}) for time update
      - Apply Potter sequential scalar updates for range and range-rate
    """

    def __init__(
        self,
        X0_star: np.ndarray,
        P0: np.ndarray,
        R: np.ndarray,
        Q: np.ndarray,
        dyn_fun,
        dyn_jac,
        station_state_map: dict | None = None,
        prop_settings: PropSettings | None = None,
        station_start_index: int = 9,
        num_stations: int = 3,
        omega_vec: np.ndarray | None = None,
        first_pass_gap_s: float = 6 * 3600.0,
    ):
        super().__init__(X0_star, P0, R, Q)

        self.X0_star = np.asarray(X0_star, dtype=float).reshape(-1).copy()
        self.n = self.X0_star.size

        self.xhat = np.zeros(self.n, dtype=float)
        self.xbar = np.zeros(self.n, dtype=float)

        self.dyn_fun = dyn_fun
        self.dyn_jac = dyn_jac

        self.station_state_map = station_state_map
        self.station_start_index = int(station_start_index)
        self.num_stations = int(num_stations)

        self.prop_settings = PropSettings() if prop_settings is None else prop_settings

        if omega_vec is None:
            self.omega_vec = np.array([0.0, 0.0, 7.2921158553e-5], dtype=float)
        else:
            self.omega_vec = np.asarray(omega_vec, dtype=float).reshape(3)

        self.first_pass_gap_s = float(first_pass_gap_s)

    def propagate_state_and_stm_history(self, t_eval: np.ndarray):
        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
        X_hist, Phi_i0_hist = propagate_x_phi_history(
            x0=self.X0_star,
            t_eval=t_eval,
            f=self.dyn_fun,
            A=self.dyn_jac,
            settings=self.prop_settings,
        )
        return X_hist, Phi_i0_hist

    @staticmethod
    def phi_i0_to_phi_step(Phi_i0_hist: np.ndarray):
        """
        Convert Phi(t_i,t0) to Phi(t_i,t_{i-1}) via:
            Phi_i_im1 = Phi_i0 @ inv(Phi_im10)
                      = Phi_i0 @ solve(Phi_im10, I)
        """
        Phi_i0_hist = np.asarray(Phi_i0_hist, dtype=float)
        N, n, _ = Phi_i0_hist.shape
        Phi_step = np.zeros_like(Phi_i0_hist)
        Phi_step[0] = np.eye(n)

        I = np.eye(n)
        for i in range(1, N):
            Phi_step[i] = Phi_i0_hist[i] @ np.linalg.solve(Phi_i0_hist[i - 1], I)

        return Phi_step

    def G(self, station_or_key, X: np.ndarray, t: float):
        """
        Measurement prediction h(X,t) -> [rho, rho_dot].
        """
        X = np.asarray(X, dtype=float).reshape(-1)
        r_sc = X[0:3]
        v_sc = X[3:6]

        if self.station_state_map is not None:
            st_idx = self.station_state_map[station_or_key]
            st_start = self.station_start_index + 3 * int(st_idx)
            r_gs = X[st_start:st_start + 3]
            v_gs = np.cross(self.omega_vec, r_gs)
        else:
            st_obj = station_or_key
            r_gs, v_gs = st_obj.station_eci(float(t))

        dr = r_sc - r_gs
        rho = np.linalg.norm(dr)
        if rho <= 0.0 or not np.isfinite(rho):
            return None
        rho_dot = float(np.dot(dr, (v_sc - v_gs)) / rho)
        if not np.isfinite(rho_dot):
            return None
        return np.array([rho, rho_dot], dtype=float)

    @staticmethod
    def _potter_update(x, P, innovation_scalar, H_row, R_scalar):
        """
        Single scalar Potter square-root update.
        """
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            P_sym = 0.5 * (P + P.T)
            P_sym += np.eye(P.shape[0]) * 1e-18
            S = np.linalg.cholesky(P_sym)

        F = S.T @ H_row.T
        inv_var = (F.T @ F) + R_scalar
        alpha = 1.0 / inv_var
        gamma = 1.0 / (1.0 + np.sqrt(R_scalar * alpha))

        K = alpha * (S @ F)
        SF = S @ F
        S_new = S - (alpha * gamma) * np.outer(SF, F)

        P_new = S_new @ S_new.T
        x_new = x + K * innovation_scalar
        return x_new, P_new

    def run(self, all_meas, stations=None, Xtrue_meas: np.ndarray | None = None):
        # -----------------------------
        # 0) Setup / sort
        # -----------------------------
        station_map = {}
        if stations is not None:
            station_map = {st.name: st for st in stations}

        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
        t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
        st_meas = [m["station"] for m in all_meas]
        N = len(all_meas)

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (N, self.n):
                raise ValueError(f"Xtrue_meas must have shape ({N}, {self.n}) aligned with sorted measurement order.")

        # -----------------------------
        # 1) Precompute reference + Phi steps
        # -----------------------------
        Xstar_hist, Phi_i0_hist = self.propagate_state_and_stm_history(t_meas)
        Phi_step = self.phi_i0_to_phi_step(Phi_i0_hist)

        # -----------------------------
        # 2) Allocate outputs
        # -----------------------------
        residuals = np.full((N, 2), np.nan, dtype=float)
        resid_pf = np.full((N, 2), np.nan, dtype=float)
        postfit_nl = np.full((N, 2), np.nan, dtype=float)

        X_pf = np.full((N, self.n), np.nan, dtype=float)
        P_meas = np.full((N, self.n, self.n), np.nan, dtype=float)
        P_pf = np.full((N, self.n * self.n), np.nan, dtype=float)
        two_sigma = np.full((N, self.n), np.nan, dtype=float)

        state_error = None if Xtrue_meas is None else np.full((N, self.n), np.nan, dtype=float)

        # -----------------------------
        # 3) Init filter vars (locals)
        # -----------------------------
        x_hat = np.zeros(self.n, dtype=float)
        P = np.array(self.P0, dtype=float).copy()

        # -----------------------------
        # 4) Main filter loop
        # -----------------------------
        for j in range(N):
            t = float(t_meas[j])
            st_key = st_meas[j]

            m = all_meas[j]
            if "rho_km" in m:
                Y = np.array([m["rho_km"] * 1000.0, m["rho_dot_km_s"] * 1000.0], dtype=float)
            else:
                Y = np.array([m["rho_m"], m["rho_dot_m_s"]], dtype=float)

            have_meas = np.isfinite(Y).all()

            Xstar = Xstar_hist[j, :]
            Phi = Phi_step[j, :, :]

            # ---- Time Update (error-state) ----
            xbar = Phi @ x_hat
            Pbar = Phi @ P @ Phi.T + self.Q

            # predicted measurement at reference
            if self.station_state_map is not None:
                st_rep = st_key
            else:
                st_rep = station_map[st_key]

            C = self.G(st_rep, Xstar, t) if have_meas else None

            if (C is not None) and np.isfinite(C).all():
                OminusC = Y - C
                residuals[j, :] = OminusC

                # measurement Jacobian
                if self.station_state_map is not None:
                    st_idx = self.station_state_map[st_key]
                    Htilde = H_tilde_range_rangerate_augmented(
                        state=Xstar,
                        stat_idx=int(st_idx),
                        omega_rad_s=float(self.omega_vec[2]),
                        station_start_index=self.station_start_index,
                        num_stations=self.num_stations,
                    )
                else:
                    r_gs, v_gs = st_rep.station_eci(t)
                    H_sc = H_range_rangerate(Xstar[:3], Xstar[3:], r_gs, v_gs)
                    Htilde = np.zeros((2, self.n), dtype=float)
                    Htilde[:, :6] = H_sc

                # Potter sequential scalar updates
                x_curr = xbar.copy()
                P_curr = Pbar.copy()
                for i in range(2):
                    inn_scalar = OminusC[i] - (Htilde[i, :] @ x_curr)
                    x_curr, P_curr = self._potter_update(
                        x_curr, P_curr, inn_scalar, Htilde[i, :], float(self.R[i, i])
                    )

                x_hat = x_curr
                P = P_curr
                resid_pf[j, :] = OminusC - (Htilde @ x_hat)
            else:
                x_hat, P = xbar, Pbar

            # post-fit state and storage
            X_post = Xstar + x_hat
            X_pf[j, :] = X_post
            P_meas[j, :, :] = P
            P_pf[j, :] = P.reshape(-1, order="F")
            two_sigma[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P), 0.0))

            # nonlinear postfit residual
            if have_meas:
                C_post = self.G(st_rep, X_post, t)
                if C_post is not None and np.isfinite(C_post).all():
                    postfit_nl[j, :] = Y - C_post

            if state_error is not None:
                state_error[j, :] = X_post - Xtrue_meas[j, :]

            self.Xhat = X_post
            self.log_epoch(
                t,
                postfit_resid=postfit_nl[j, :],
                Xtrue=(None if Xtrue_meas is None else Xtrue_meas[j, :]),
            )

        self.xhat = x_hat
        self.Phat = P
        self.Xhat = X_pf[-1, :]

        rms_final = self.print_rms_summary(label="POTTER-LKF18", first_pass_gap_s=self.first_pass_gap_s)

        return {
            "t_meas": t_meas,
            "station_meas": st_meas,
            "xhat_meas": X_pf,
            "Xhat_meas": X_pf,
            "X_pf": X_pf,
            "prefit_resids_final": residuals,
            "postfit_resids_linear_final": resid_pf,
            "postfit_resids_meas": postfit_nl,
            "P_meas": P_meas,
            "Phat_meas": P_meas,
            "P_pf": P_pf,
            "two_sigma_meas": two_sigma,
            "state_error_meas": state_error,
            "rms_final": rms_final,
            "rms_by_iter": None,
            "R": self.R,
        }


def potter_post_process_18(
    *,
    out: dict,
    length_unit_in: str = "m",
    length_unit_out: str = "m",
):
    """
    Convenience wrapper so Potter output can use the existing plotting pipeline.
    """
    from src.helpers.plotting.post_processing import run_filter_post_processing_18

    return run_filter_post_processing_18(
        out=out,
        length_unit_in=length_unit_in,
        length_unit_out=length_unit_out,
    )
