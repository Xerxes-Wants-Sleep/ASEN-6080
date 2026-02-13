"""
filter_18_state.py

This module provides two Kalman filter implementations for an 18-state orbit-determination problem
with range / range-rate measurements:

State (n = 18) is assumed to be ordered as:
    x = [ r_sc(3), v_sc(3), mu(1), J2(1), CD(1), r_s1(3), r_s2(3), r_s3(3) ]

Measurement model (per measurement epoch, per station):
    y = [ rho, rho_dot ]
where
    rho     = || r_sc - r_gs ||
    rho_dot = (r_sc - r_gs) · (v_sc - v_gs) / rho

If the station positions are included in the state (18-state mode), then:
    r_gs is taken directly from the state vector,
    v_gs is computed as v_gs = ω × r_gs  (Earth rotation in inertial frame).

Filters included:
- LinearizedKalmanFilter18State (LKF / CKF-style error-state filter)
    * Propagates a single reference trajectory X*(t) and STM history once.
    * Propagates the error-state with step STMs and applies linearized measurement updates.

- ExtendedKalmanFilter18State (EKF in state form)
    * Propagates the current estimate and STM step-by-step between measurements.
    * Linearizes measurements about the predicted state and updates the full state.

Units:
- This code standardizes measurement dict inputs to meters and m/s if you provide:
    {rho_km, rho_dot_km_s}  -> converted to (m, m/s)
  or directly uses:
    {rho_m, rho_dot_m_s}    -> used as-is
Make sure your dynamics + state units are consistent with the standardized measurement units.
"""

from __future__ import annotations

import numpy as np

from src.Functions.propagation import propagate_x_phi_history, PropSettings
from src.Functions.range_rangerate import H_range_rangerate, H_tilde_range_rangerate_augmented


class KalmanFilterBase:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, R: np.ndarray, Q: np.ndarray):
        self.Xhat = np.array(x0, dtype=float).copy()
        self.Phat = np.array(P0, dtype=float).copy()
        self.R = np.array(R, dtype=float).copy()   # (2,2)
        self.Q = np.array(Q, dtype=float).copy()   # (n,n)
        self.P0 = np.array(P0, dtype=float).copy()

        self.hist = {
            "t": [],
            "state_err": [],   # Xhat - Xtrue (if provided)
            "postfit": [],     # Y - h(Xhat,t) (NaNs allowed)
        }

    def log_epoch(self, t, postfit_resid, Xtrue=None):
        self.hist["t"].append(float(t))
        self.hist["postfit"].append(np.array(postfit_resid, dtype=float).reshape(-1))
        if Xtrue is not None:
            e = self.Xhat.reshape(-1) - np.array(Xtrue, dtype=float).reshape(-1)
            self.hist["state_err"].append(e)

    def rms_nan(self, A, axis=0):
        return np.sqrt(np.nanmean(A**2, axis=axis))

    def compute_first_pass_mask(self, t, gap_s=None):
        # keep consistent with your current filters.py behavior: ignore first measurement only
        t = np.asarray(t, dtype=float).reshape(-1)
        keep_all = np.ones_like(t, dtype=bool)
        keep_ignore_first = np.ones_like(t, dtype=bool)
        if t.size > 0:
            keep_ignore_first[0] = False
        return keep_all, keep_ignore_first

    def compute_rms_summary(self, first_pass_gap_s=6 * 3600.0):
        t = np.asarray(self.hist["t"], dtype=float)
        keep_all, keep_ignore_first = self.compute_first_pass_mask(t, first_pass_gap_s)

        r = np.vstack(self.hist["postfit"]) if len(self.hist["postfit"]) else np.empty((0, 0))
        rms_post_all = self.rms_nan(r[keep_all, :], axis=0) if r.size else None
        rms_post_ign = self.rms_nan(r[keep_ignore_first, :], axis=0) if r.size else None
        rms_post_norm_all = None
        rms_post_norm_ign = None

        if r.size and self.R is not None:
            Rinv = np.linalg.inv(self.R)
            r_all = r[keep_all, :]
            r_ign = r[keep_ignore_first, :]
            norm_all = np.einsum("ij,jk,ik->i", r_all, Rinv, r_all)
            norm_ign = np.einsum("ij,jk,ik->i", r_ign, Rinv, r_ign)
            rms_post_norm_all = float(np.sqrt(np.nanmean(norm_all)))
            rms_post_norm_ign = float(np.sqrt(np.nanmean(norm_ign)))

        state_comp_all = state_comp_ign = None
        pos3_all = pos3_ign = None
        vel3_all = vel3_ign = None

        if len(self.hist["state_err"]) > 0:
            e = np.vstack(self.hist["state_err"])
            state_comp_all = self.rms_nan(e[keep_all, :], axis=0)
            state_comp_ign = self.rms_nan(e[keep_ignore_first, :], axis=0)

            # only compute these if the state has at least 6 components
            if e.shape[1] >= 6:
                pos3_all = float(self.rms_nan(np.linalg.norm(e[keep_all, 0:3], axis=1), axis=0))
                pos3_ign = float(self.rms_nan(np.linalg.norm(e[keep_ignore_first, 0:3], axis=1), axis=0))
                vel3_all = float(self.rms_nan(np.linalg.norm(e[keep_all, 3:6], axis=1), axis=0))
                vel3_ign = float(self.rms_nan(np.linalg.norm(e[keep_ignore_first, 3:6], axis=1), axis=0))

        return {
            "keep_all_mask": keep_all,
            "keep_ignore_first_mask": keep_ignore_first,
            "state_comp_all": state_comp_all,
            "state_comp_ignore_first": state_comp_ign,
            "pos3_all": pos3_all,
            "pos3_ignore_first": pos3_ign,
            "vel3_all": vel3_all,
            "vel3_ignore_first": vel3_ign,
            "postfit_all": rms_post_all,
            "postfit_ignore_first": rms_post_ign,
            "postfit_norm_all": rms_post_norm_all,
            "postfit_norm_ignore_first": rms_post_norm_ign,
        }

    def print_rms_summary(self, label="", first_pass_gap_s=6 * 3600.0):
        rms = self.compute_rms_summary(first_pass_gap_s=first_pass_gap_s)
        hdr = f"\n===== RMS SUMMARY {label} =====" if label else "\n===== RMS SUMMARY ====="
        print(hdr)

        if rms["state_comp_all"] is not None:
            print("State error RMS (component-wise) [all]:")
            print(rms["state_comp_all"])
            print("State error RMS (component-wise) [ignore first]:")
            print(rms["state_comp_ignore_first"])
            if rms["pos3_all"] is not None:
                print(f"Pos3 RMS all / ignore: {rms['pos3_all']:.6g} / {rms['pos3_ignore_first']:.6g}")
                print(f"Vel3 RMS all / ignore: {rms['vel3_all']:.6g} / {rms['vel3_ignore_first']:.6g}")
        else:
            print("State error RMS: (truth not logged)")

        if rms["postfit_all"] is not None:
            if rms.get("postfit_norm_all") is not None:
                print(
                    "Postfit residual RMS [all]: "
                    f"RMS rho = {rms['postfit_all'][0]:.6g} | "
                    f"RMS rhodot = {rms['postfit_all'][1]:.6g} | "
                    f"RMS norm = {rms['postfit_norm_all']:.6g}"
                )
                print(
                    "Postfit residual RMS [ignore first]: "
                    f"RMS rho = {rms['postfit_ignore_first'][0]:.6g} | "
                    f"RMS rhodot = {rms['postfit_ignore_first'][1]:.6g} | "
                    f"RMS norm = {rms['postfit_norm_ignore_first']:.6g}"
                )
            else:
                print("Postfit residual RMS [all]:")
                print(rms["postfit_all"])
                print("Postfit residual RMS [ignore first]:")
                print(rms["postfit_ignore_first"])
        else:
            print("Postfit residual RMS: (no residuals logged)")

        return rms

    @staticmethod
    def compute_rms_summary_from_arrays(t, postfit, state_err=None, first_pass_gap_s=6 * 3600.0, R=None):
        t = np.asarray(t, dtype=float).reshape(-1)

        if t.size < 2:
            keep_all = np.ones_like(t, dtype=bool)
            keep_ignore_first = np.ones_like(t, dtype=bool)
        else:
            dt = np.diff(t)
            idx_gap = np.where(dt > first_pass_gap_s)[0]
            keep_all = np.ones_like(t, dtype=bool)
            keep_ignore_first = np.ones_like(t, dtype=bool)
            if idx_gap.size > 0:
                end_first_pass = idx_gap[0]
                keep_ignore_first[: end_first_pass + 1] = False

        def rms_nan(A, axis=0):
            return np.sqrt(np.nanmean(A**2, axis=axis))

        postfit = np.asarray(postfit, dtype=float)
        rms_post_all = rms_nan(postfit[keep_all, :], axis=0) if postfit.size else None
        rms_post_ign = rms_nan(postfit[keep_ignore_first, :], axis=0) if postfit.size else None
        rms_post_norm_all = None
        rms_post_norm_ign = None

        if postfit.size and R is not None:
            Rinv = np.linalg.inv(np.asarray(R, dtype=float))
            pf_all = postfit[keep_all, :]
            pf_ign = postfit[keep_ignore_first, :]
            norm_all = np.einsum("ij,jk,ik->i", pf_all, Rinv, pf_all)
            norm_ign = np.einsum("ij,jk,ik->i", pf_ign, Rinv, pf_ign)
            rms_post_norm_all = float(np.sqrt(np.nanmean(norm_all)))
            rms_post_norm_ign = float(np.sqrt(np.nanmean(norm_ign)))

        state_comp_all = state_comp_ign = None
        pos3_all = pos3_ign = None
        vel3_all = vel3_ign = None

        if state_err is not None:
            e = np.asarray(state_err, dtype=float)
            if e.size:
                state_comp_all = rms_nan(e[keep_all, :], axis=0)
                state_comp_ign = rms_nan(e[keep_ignore_first, :], axis=0)

                if e.shape[1] >= 6:
                    pos3_all = float(rms_nan(np.linalg.norm(e[keep_all, 0:3], axis=1), axis=0))
                    pos3_ign = float(rms_nan(np.linalg.norm(e[keep_ignore_first, 0:3], axis=1), axis=0))
                    vel3_all = float(rms_nan(np.linalg.norm(e[keep_all, 3:6], axis=1), axis=0))
                    vel3_ign = float(rms_nan(np.linalg.norm(e[keep_ignore_first, 3:6], axis=1), axis=0))

        return {
            "keep_all_mask": keep_all,
            "keep_ignore_first_mask": keep_ignore_first,
            "state_comp_all": state_comp_all,
            "state_comp_ignore_first": state_comp_ign,
            "pos3_all": pos3_all,
            "pos3_ignore_first": pos3_ign,
            "vel3_all": vel3_all,
            "vel3_ignore_first": vel3_ign,
            "postfit_all": rms_post_all,
            "postfit_ignore_first": rms_post_ign,
            "postfit_norm_all": rms_post_norm_all,
            "postfit_norm_ignore_first": rms_post_norm_ign,
        }

    def reset_history(self):
        self.hist = {"t": [], "state_err": [], "postfit": []}


class LinearizedKalmanFilter18State(KalmanFilterBase):
    """
    LKF in error-state form about a precomputed reference trajectory X*(t).

    Workflow:
      1) Propagate reference state and STM history once: X*(t_i), Phi(t_i, t0)
      2) Convert to step STMs: Phi(t_i, t_{i-1})
      3) Propagate the error-state and covariance with the step STM
      4) Update using linearized measurement model about X*(t_i)
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
        recenter_reference: bool = False,  # kept for API parity; default False
    ):
        super().__init__(X0_star, P0, R, Q)

        self.X0_star = np.asarray(X0_star, dtype=float).reshape(-1).copy()
        self.n = self.X0_star.size

        self.xhat = np.zeros(self.n, dtype=float)  # estimated error state
        self.xbar = np.zeros(self.n, dtype=float)  # predicted error state

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
        self.recenter_reference = bool(recenter_reference)

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

        - If station_state_map is provided (18-state mode): station_or_key is the measurement's station key,
          and station inertial position is read from the state vector.
        - Otherwise: station_or_key is assumed to be a station object with station_eci(t)->(r,v).
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

    # ------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------
    def run(self, all_meas, stations=None, Xtrue_meas: np.ndarray | None = None):
        # station map (only used if station_state_map is None)
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

        # precompute reference + stm steps
        Xstar_hist, Phi_i0_hist = self.propagate_state_and_stm_history(t_meas)
        Phi_step = self.phi_i0_to_phi_step(Phi_i0_hist)

        # allocate outputs
        residuals = np.full((N, 2), np.nan, dtype=float)
        resid_pf = np.full((N, 2), np.nan, dtype=float)
        postfit_nl = np.full((N, 2), np.nan, dtype=float)

        X_pf = np.full((N, self.n), np.nan, dtype=float)
        P_meas = np.full((N, self.n, self.n), np.nan, dtype=float)
        P_pf = np.full((N, self.n * self.n), np.nan, dtype=float)
        two_sigma = np.full((N, self.n), np.nan, dtype=float)

        state_error = None if Xtrue_meas is None else np.full((N, self.n), np.nan, dtype=float)

        x_hat = np.zeros(self.n, dtype=float)
        P = np.array(self.P0, dtype=float).copy()
        I_n = np.eye(self.n)
        I_2 = np.eye(2)

        for j in range(N):
            t = float(t_meas[j])
            st_key = st_meas[j]

            # observed measurement (standardize to meters + m/s if given in km)
            m = all_meas[j]
            if "rho_km" in m:
                Y = np.array([m["rho_km"] * 1000.0, m["rho_dot_km_s"] * 1000.0], dtype=float)
            else:
                Y = np.array([m["rho_m"], m["rho_dot_m_s"]], dtype=float)

            Xstar = Xstar_hist[j, :]
            Phi = Phi_step[j, :, :]

            # time update
            xbar = Phi @ x_hat
            Pbar = Phi @ P @ Phi.T + self.Q

            have_meas = np.isfinite(Y).all()

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
                    H_sc = H_range_rangerate(Xstar[:3], Xstar[3:], r_gs, v_gs)  # (2,6)
                    Htilde = np.zeros((2, self.n), dtype=float)
                    Htilde[:, :6] = H_sc

                # Kalman gain
                S = Htilde @ Pbar @ Htilde.T + self.R
                K = Pbar @ Htilde.T @ np.linalg.solve(S, I_2)

                # measurement update (error-state form)
                x_hat = xbar + K @ (OminusC - Htilde @ xbar)

                A = I_n - K @ Htilde
                P = A @ Pbar @ A.T + K @ self.R @ K.T  # Joseph

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

        rms_final = self.print_rms_summary(label="LKF18", first_pass_gap_s=self.first_pass_gap_s)

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
            "Phat_meas": P_meas,  # alias (helps warmstart like your existing LKF)
            "P_pf": P_pf,
            "two_sigma_meas": two_sigma,
            "state_error_meas": state_error,
            "rms_final": rms_final,
            "rms_by_iter": None,
            "R": self.R,
        }


class ExtendedKalmanFilter18State(KalmanFilterBase):
    """
    EKF in state form:

      - Propagate Xhat (nonlinear dynamics) from t_{k-1} -> t_k to get Xbar
      - Propagate covariance using STM Phi(t_k, t_{k-1}) from the same propagation
      - Linearize measurement about Xbar and update:
            Xhat = Xbar + K (y - h(Xbar))
    """

    def __init__(
        self,
        x0: np.ndarray,
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
        super().__init__(x0, P0, R, Q)

        self.Xhat = np.asarray(self.Xhat, dtype=float).reshape(-1)
        self.Phat = np.asarray(self.Phat, dtype=float)

        self.n = self.Xhat.size
        if self.Phat.shape != (self.n, self.n):
            raise ValueError(f"P0 must be ({self.n},{self.n}); got {self.Phat.shape}")

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

    @staticmethod
    def print_rms_summary_dict(rms: dict, label: str = ""):
        hdr = f"\n===== RMS SUMMARY {label} =====" if label else "\n===== RMS SUMMARY ====="
        print(hdr)

        if rms.get("state_comp_all") is not None:
            print("State error RMS (component-wise) [all]:")
            print(rms["state_comp_all"])
            print("State error RMS (component-wise) [ignore first]:")
            print(rms["state_comp_ignore_first"])
            if rms.get("pos3_all") is not None:
                print(f"Pos3 RMS all / ignore: {rms['pos3_all']:.6g} / {rms['pos3_ignore_first']:.6g}")
                print(f"Vel3 RMS all / ignore: {rms['vel3_all']:.6g} / {rms['vel3_ignore_first']:.6g}")
        else:
            print("State error RMS: (truth not provided)")

        if rms.get("postfit_all") is not None:
            print("Postfit residual RMS [all]:")
            print(rms["postfit_all"])
            print("Postfit residual RMS [ignore first]:")
            print(rms["postfit_ignore_first"])
        else:
            print("Postfit residual RMS: (no residuals)")

        return rms

    def G(self, station_or_key, X: np.ndarray, t: float):
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

    def _propagate_state_and_stm_step(self, t0: float, t1: float, X0: np.ndarray):
        """
        Propagate state and STM from t0->t1 with initial STM = I.
        Uses propagate_x_phi_history over a 2-point grid so that Phi(t1,t0) is returned directly.
        """
        t0 = float(t0)
        t1 = float(t1)
        X0 = np.asarray(X0, dtype=float).reshape(-1)

        if np.isclose(t1, t0):
            return X0.copy(), np.eye(self.n)

        t_eval = np.array([t0, t1], dtype=float)
        X_hist, Phi_hist = propagate_x_phi_history(
            x0=X0,
            t_eval=t_eval,
            f=self.dyn_fun,
            A=self.dyn_jac,
            settings=self.prop_settings,
        )

        X1 = X_hist[-1, :].copy()
        Phi_10 = Phi_hist[-1, :, :].copy()  # Phi(t1,t0)

        return X1, Phi_10

    def run(self, all_meas, stations=None, Xtrue_meas=None, t_prev_init=None):
        station_map = {}
        if stations is not None:
            station_map = {st.name: st for st in stations}

        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
        N = len(all_meas)

        t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
        st_meas = [m["station"] for m in all_meas]

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (N, self.n):
                raise ValueError(f"Xtrue_meas must have shape ({N}, {self.n}) aligned with sorted measurement order.")

        residuals = np.full((N, 2), np.nan, dtype=float)
        resid_pf = np.full((N, 2), np.nan, dtype=float)
        postfit_nl = np.full((N, 2), np.nan, dtype=float)

        X_pf = np.full((N, self.n), np.nan, dtype=float)
        P_meas = np.full((N, self.n, self.n), np.nan, dtype=float)
        P_pf = np.full((N, self.n * self.n), np.nan, dtype=float)
        two_sigma = np.full((N, self.n), np.nan, dtype=float)

        state_error = None if Xtrue_meas is None else np.full((N, self.n), np.nan, dtype=float)

        X_hat = np.asarray(self.Xhat, dtype=float).reshape(self.n,)
        P = np.asarray(self.Phat, dtype=float).reshape(self.n, self.n)

        prev_time = float(t_meas[0]) if t_prev_init is None else float(t_prev_init)

        I_n = np.eye(self.n)
        I_2 = np.eye(2)

        for j in range(N):
            t = float(t_meas[j])
            st_key = st_meas[j]

            m = all_meas[j]
            if "rho_km" in m:
                Y = np.array([m["rho_km"] * 1000.0, m["rho_dot_km_s"] * 1000.0], dtype=float)
            else:
                Y = np.array([m["rho_m"], m["rho_dot_m_s"]], dtype=float)

            have_meas = np.isfinite(Y).all()

            # propagate
            Xbar, Phi = self._propagate_state_and_stm_step(prev_time, t, X_hat)
            Pbar = Phi @ P @ Phi.T + self.Q

            if self.station_state_map is not None:
                st_rep = st_key
            else:
                st_rep = station_map[st_key]

            C = self.G(st_rep, Xbar, t) if have_meas else None

            if (C is not None) and np.isfinite(C).all():
                OminusC = Y - C
                residuals[j, :] = OminusC

                if self.station_state_map is not None:
                    st_idx = self.station_state_map[st_key]
                    Htilde = H_tilde_range_rangerate_augmented(
                        state=Xbar,
                        stat_idx=int(st_idx),
                        omega_rad_s=float(self.omega_vec[2]),
                        station_start_index=self.station_start_index,
                        num_stations=self.num_stations,
                    )
                else:
                    r_gs, v_gs = st_rep.station_eci(t)
                    H_sc = H_range_rangerate(Xbar[:3], Xbar[3:], r_gs, v_gs)
                    Htilde = np.zeros((2, self.n), dtype=float)
                    Htilde[:, :6] = H_sc

                S = Htilde @ Pbar @ Htilde.T + self.R
                K = Pbar @ Htilde.T @ np.linalg.solve(S, I_2)

                # EKF state update
                X_hat = Xbar + K @ OminusC

                A = I_n - K @ Htilde
                P = A @ Pbar @ A.T + K @ self.R @ K.T  # Joseph

                # linear post-fit residual around Xbar
                dx = X_hat - Xbar
                resid_pf[j, :] = OminusC - (Htilde @ dx)

                # nonlinear post-fit
                C_hat = self.G(st_rep, X_hat, t)
                if C_hat is not None and np.isfinite(C_hat).all():
                    postfit_nl[j, :] = Y - C_hat
            else:
                X_hat, P = Xbar, Pbar

            X_pf[j, :] = X_hat
            P_meas[j, :, :] = P
            P_pf[j, :] = P.reshape(-1, order="F")
            two_sigma[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P), 0.0))

            if state_error is not None:
                state_error[j, :] = X_hat - Xtrue_meas[j, :]

            prev_time = t

            self.Xhat = X_hat
            self.log_epoch(
                t,
                postfit_resid=postfit_nl[j, :],
                Xtrue=(None if Xtrue_meas is None else Xtrue_meas[j, :]),
            )

        self.Xhat = X_hat
        self.Phat = P

        rms_final = self.print_rms_summary(label="EKF18", first_pass_gap_s=self.first_pass_gap_s)

        return {
            "t_meas": t_meas,
            "station_meas": st_meas,
            "xhat_meas": X_pf,
            "Xhat_meas": X_pf,
            "X_pf": X_pf,
            "state_error_meas": state_error,
            "prefit_resids_final": residuals,
            "postfit_resids_linear_final": resid_pf,
            "postfit_resids_meas": postfit_nl,
            "P_meas": P_meas,
            "P_pf": P_pf,
            "two_sigma_meas": two_sigma,
            "rms_final": rms_final,
            "rms_by_iter": None,
            "R": self.R,
        }

    def run_warmstarted(
        self,
        all_meas,
        stations,
        lkf: "LinearizedKalmanFilter18State",
        num_init_meas: int = 100,
        Xtrue_meas: np.ndarray | None = None,
    ):
        """
        Warm-start EKF using LKF on the first num_init_meas observations, then continue EKF
        from that posterior (state + covariance) forward.

        Returns a SINGLE combined output dict matching your usual plotting pipeline keys.
        """
        all_meas_sorted = sorted(all_meas, key=lambda m: float(m["t"]))
        mcount = len(all_meas_sorted)

        if mcount == 0:
            return {
                "lkf_init": None,
                "ekf": None,
                "t_meas": np.array([], dtype=float),
                "station_meas": [],
                "Xhat_meas": np.empty((0, self.n), dtype=float),
                "xhat_meas": np.empty((0, self.n), dtype=float),
                "P_meas": np.empty((0, self.n, self.n), dtype=float),
                "two_sigma_meas": np.empty((0, self.n), dtype=float),
                "state_error_meas": None,
                "prefit_resids_final": np.empty((0, 2), dtype=float),
                "postfit_resids_linear_final": np.empty((0, 2), dtype=float),
                "postfit_resids_meas": np.empty((0, 2), dtype=float),
                "P_pf": np.empty((0, self.n * self.n), dtype=float),
                "rms_final": None,
                "rms_by_iter": None,
            }

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (mcount, self.n):
                raise ValueError(f"Xtrue_meas must have shape ({mcount}, {self.n}) aligned with sorted measurement order.")

        Ninit = int(num_init_meas)
        if Ninit <= 0:
            return self.run(all_meas_sorted, stations, Xtrue_meas=Xtrue_meas, t_prev_init=None)

        Ninit = min(Ninit, mcount)
        init_meas = all_meas_sorted[:Ninit]
        rest_meas = all_meas_sorted[Ninit:]

        Xtrue_init = Xtrue_rest = None
        if Xtrue_meas is not None:
            Xtrue_init = Xtrue_meas[:Ninit, :]
            Xtrue_rest = Xtrue_meas[Ninit:, :]

        if hasattr(lkf, "reset_history"):
            lkf.reset_history()
        if hasattr(self, "reset_history"):
            self.reset_history()

        # 1) LKF init chunk
        lkf_out = lkf.run(init_meas, stations, Xtrue_meas=Xtrue_init)
        X_start = np.asarray(lkf_out["Xhat_meas"][-1], dtype=float).reshape(self.n,)
        P_hist = lkf_out.get("P_meas", lkf_out.get("Phat_meas", None))
        if P_hist is None:
            raise KeyError("LKF output missing P_meas/Phat_meas needed for warmstart.")
        P_start = np.asarray(P_hist[-1], dtype=float).reshape(self.n, self.n)
        t_start = float(lkf_out["t_meas"][-1])

        # 2) seed EKF
        self.Xhat = X_start.copy()
        self.Phat = P_start.copy()

        if len(rest_meas) == 0:
            out_comb = dict(lkf_out)
            out_comb["xhat_meas"] = out_comb.get("xhat_meas", out_comb["Xhat_meas"])
            out_comb["rms_by_iter"] = None
            out_comb["lkf_init"] = lkf_out
            out_comb["ekf"] = None
            out_comb["t_start_ekf"] = t_start
            return out_comb

        # 3) EKF remainder
        ekf_out = self.run(rest_meas, stations, Xtrue_meas=Xtrue_rest, t_prev_init=t_start)

        # 4) stitch
        t_comb = np.hstack([lkf_out["t_meas"], ekf_out["t_meas"]])
        station_comb = list(lkf_out["station_meas"]) + list(ekf_out["station_meas"])
        Xhat_comb = np.vstack([lkf_out["Xhat_meas"], ekf_out["Xhat_meas"]])

        P_meas_lkf = lkf_out.get("P_meas", lkf_out.get("Phat_meas"))
        P_meas_ekf = ekf_out.get("P_meas", None)
        if P_meas_ekf is None:
            raise KeyError("EKF output missing P_meas needed for combined plots.")
        P_meas_comb = np.concatenate([P_meas_lkf, P_meas_ekf], axis=0)

        two_sigma_comb = np.vstack([lkf_out["two_sigma_meas"], ekf_out["two_sigma_meas"]])
        prefit_comb = np.vstack([lkf_out["prefit_resids_final"], ekf_out["prefit_resids_final"]])
        postfit_lin_comb = np.vstack([lkf_out["postfit_resids_linear_final"], ekf_out["postfit_resids_linear_final"]])
        postfit_nl_comb = np.vstack([lkf_out["postfit_resids_meas"], ekf_out["postfit_resids_meas"]])

        Ppf_lkf = lkf_out.get("P_pf", None)
        Ppf_ekf = ekf_out.get("P_pf", None)
        P_pf_comb = None if (Ppf_lkf is None or Ppf_ekf is None) else np.vstack([Ppf_lkf, Ppf_ekf])

        if Xtrue_meas is None:
            err_comb = None
        else:
            err_comb = np.vstack([lkf_out["state_error_meas"], ekf_out["state_error_meas"]])

        rms_combined = KalmanFilterBase.compute_rms_summary_from_arrays(
            t=t_comb,
            postfit=postfit_nl_comb,
            state_err=err_comb,
            first_pass_gap_s=self.first_pass_gap_s,
            R=self.R,
        )

        return {
            "lkf_init": lkf_out,
            "ekf": ekf_out,
            "t_start_ekf": t_start,
            "t_meas": t_comb,
            "station_meas": station_comb,
            "Xhat_meas": Xhat_comb,
            "xhat_meas": Xhat_comb,
            "P_meas": P_meas_comb,
            "two_sigma_meas": two_sigma_comb,
            "state_error_meas": err_comb,
            "prefit_resids_final": prefit_comb,
            "postfit_resids_linear_final": postfit_lin_comb,
            "postfit_resids_meas": postfit_nl_comb,
            "P_pf": P_pf_comb,
            "rms_final": rms_combined,
            "rms_by_iter": None,
            "R": self.R,
        }
