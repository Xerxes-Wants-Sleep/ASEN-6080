from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .jacobians import stm
from .propagation import PropSettings, propagate_x_phi_history, propagate_x_phi_step
from .snc import state_noise_compensation


class KalmanFilterBase:
    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        R: np.ndarray,
        Q: np.ndarray,
        state_mapping_dict: dict | None = None,
    ):
        self.Xhat = np.array(x0, dtype=float).reshape(-1).copy()
        self.Phat = np.array(P0, dtype=float).copy()
        self.R = np.array(R, dtype=float).copy()
        self.Q = np.array(Q, dtype=float).copy()
        self.P0 = np.array(P0, dtype=float).copy()

        self.n = int(self.Xhat.size)


        self.state_mapping_dict = {} if state_mapping_dict is None else dict(state_mapping_dict)
        self.pos_idx = self._parse_index_list(self.state_mapping_dict.get("pos_idx"), "pos_idx", required_len=3)
        self.vel_idx = self._parse_index_list(self.state_mapping_dict.get("vel_idx"), "vel_idx", required_len=3)

        self.hist = {"t": [], "state_err": [], "postfit": []}

    def _snc_covariance(self, dt: float) -> np.ndarray:
        dt = float(dt)
        return state_noise_compensation(dt, self.n, int(self.Q.shape[0]), self.Q)

    def _parse_index_list(self, idx, name: str, required_len: int | None = None):
        if idx is None:
            return None
        if isinstance(idx, slice):
            idx_list = list(range(self.n))[idx]
        else:
            idx_list = list(np.asarray(idx, dtype=int).reshape(-1))
        if required_len is not None and len(idx_list) != required_len:
            raise ValueError(f"{name} must specify {required_len} indices.")
        for i in idx_list:
            if i < 0 or i >= self.n:
                raise ValueError(f"{name} index {i} out of bounds for state size {self.n}.")
        return idx_list

    def log_epoch(self, t, postfit_resid, Xtrue=None):
        self.hist["t"].append(float(t))
        self.hist["postfit"].append(np.array(postfit_resid, dtype=float).reshape(-1))

        if Xtrue is not None:
            e = self.Xhat.reshape(-1) - np.array(Xtrue, dtype=float).reshape(-1)
            self.hist["state_err"].append(e)

    def rms_nan(self, A, axis=0):
        return np.sqrt(np.nanmean(A**2, axis=axis))

    def compute_first_pass_mask(self, t, gap_s=None):
        t = np.asarray(t, dtype=float).reshape(-1)

        keep_all = np.ones_like(t, dtype=bool)
        keep_ignore_first = np.ones_like(t, dtype=bool)
        if t.size > 0:
            keep_ignore_first[0] = False

        return keep_all, keep_ignore_first

    def _pos_vel_indices(self, n: int):
        pos_idx = self.pos_idx
        vel_idx = self.vel_idx
        if pos_idx is None or vel_idx is None:
            return None, None
        return pos_idx, vel_idx

    def compute_rms_summary(self, first_pass_gap_s=6 * 3600.0):
        t = np.asarray(self.hist["t"], dtype=float)
        keep_all, keep_ignore_first = self.compute_first_pass_mask(t, first_pass_gap_s)

        r = np.vstack(self.hist["postfit"]) if len(self.hist["postfit"]) else np.empty((0, 0))
        rms_post_all = self.rms_nan(r[keep_all, :], axis=0) if r.size else None
        rms_post_ign = self.rms_nan(r[keep_ignore_first, :], axis=0) if r.size else None

        state_comp_all = state_comp_ign = None
        pos3_all = pos3_ign = None
        vel3_all = vel3_ign = None

        if len(self.hist["state_err"]) > 0:
            e = np.vstack(self.hist["state_err"])
            state_comp_all = self.rms_nan(e[keep_all, :], axis=0)
            state_comp_ign = self.rms_nan(e[keep_ignore_first, :], axis=0)

            pos_idx, vel_idx = self._pos_vel_indices(e.shape[1])
            if pos_idx is not None and vel_idx is not None:
                pos3_all = float(self.rms_nan(np.linalg.norm(e[keep_all][:, pos_idx], axis=1), axis=0))
                pos3_ign = float(self.rms_nan(np.linalg.norm(e[keep_ignore_first][:, pos_idx], axis=1), axis=0))
                vel3_all = float(self.rms_nan(np.linalg.norm(e[keep_all][:, vel_idx], axis=1), axis=0))
                vel3_ign = float(self.rms_nan(np.linalg.norm(e[keep_ignore_first][:, vel_idx], axis=1), axis=0))

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
        }

    def print_rms_summary(self, label="", first_pass_gap_s=6 * 3600.0):
        rms = self.compute_rms_summary(first_pass_gap_s=first_pass_gap_s)

        hdr = f"\n===== RMS SUMMARY {label} =====" if label else "\n===== RMS SUMMARY ====="
        print(hdr)

        if rms["state_comp_all"] is not None:
            print("State error RMS (component-wise) [all]:")
            print(rms["state_comp_all"])
            print("State error RMS (component-wise) [ignore first pass]:")
            print(rms["state_comp_ignore_first"])
            if rms["pos3_all"] is not None:
                print(f"Pos3 RMS all / ignore: {rms['pos3_all']:.6g} / {rms['pos3_ignore_first']:.6g}")
                print(f"Vel3 RMS all / ignore: {rms['vel3_all']:.6g} / {rms['vel3_ignore_first']:.6g}")
        else:
            print("State error RMS: (truth not logged)")

        if rms["postfit_all"] is not None:
            print("Postfit residual RMS [all]:")
            print(rms["postfit_all"])
            print("Postfit residual RMS [ignore first pass]:")
            print(rms["postfit_ignore_first"])
        else:
            print("Postfit residual RMS: (no residuals logged)")

        return rms

    @staticmethod
    def compute_rms_summary_from_arrays(
        t,
        postfit,
        state_err=None,
        first_pass_gap_s=6 * 3600.0,
        state_mapping_dict: dict | None = None,
    ):
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

        state_comp_all = state_comp_ign = None
        pos3_all = pos3_ign = None
        vel3_all = vel3_ign = None

        if state_err is not None:
            e = np.asarray(state_err, dtype=float)
            if e.size:
                state_comp_all = rms_nan(e[keep_all, :], axis=0)
                state_comp_ign = rms_nan(e[keep_ignore_first, :], axis=0)

                n = e.shape[1]
                mapping = {} if state_mapping_dict is None else dict(state_mapping_dict)

                def parse(idx, required_len):
                    if idx is None:
                        return None
                    if isinstance(idx, slice):
                        idx_list = list(range(n))[idx]
                    else:
                        idx_list = list(np.asarray(idx, dtype=int).reshape(-1))
                    if len(idx_list) != required_len:
                        raise ValueError(f"RMS mapping requires {required_len} indices.")
                    return idx_list

                pos_idx = parse(mapping.get("pos_idx", None), 3) if mapping.get("pos_idx", None) is not None else None
                vel_idx = parse(mapping.get("vel_idx", None), 3) if mapping.get("vel_idx", None) is not None else None

                if pos_idx is not None and vel_idx is not None:
                    pos3_all = float(rms_nan(np.linalg.norm(e[keep_all][:, pos_idx], axis=1), axis=0))
                    pos3_ign = float(rms_nan(np.linalg.norm(e[keep_ignore_first][:, pos_idx], axis=1), axis=0))
                    vel3_all = float(rms_nan(np.linalg.norm(e[keep_all][:, vel_idx], axis=1), axis=0))
                    vel3_ign = float(rms_nan(np.linalg.norm(e[keep_ignore_first][:, vel_idx], axis=1), axis=0))

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
        }

    def reset_history(self):
        self.hist = {"t": [], "state_err": [], "postfit": []}


class LinearizedKalmanFilter(KalmanFilterBase):
    """
    LKF in error-state form about a precomputed reference trajectory X*(t).

    Reference and STM are propagated ONCE:
        X*(t_i), Phi(t_i,t0)

    Then step STM is formed by matrix math:
        Phi(t_i,t_{i-1}) = Phi(t_i,t0) @ inv(Phi(t_{i-1},t0))
                         = Phi_i0 @ solve(Phi_im10, I)
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
        state_mapping_dict: dict | None = None,
        dyn_fun=None,
        dyn_jac=None,
        prop_settings: PropSettings | None = None,
    ):
        super().__init__(X0_star, P0, R, Q, state_mapping_dict=state_mapping_dict)

        self.X0_star = np.asarray(X0_star, dtype=float).reshape(-1).copy()
        self.n = int(self.X0_star.size)

        self.xhat = np.zeros(self.n, dtype=float)
        self.xbar = np.zeros(self.n, dtype=float)

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

        self.dyn_fun = dyn_fun
        self.dyn_jac = dyn_jac
        self.use_generic_propagation = (dyn_fun is not None) or (dyn_jac is not None)
        if self.use_generic_propagation and (dyn_fun is None or dyn_jac is None):
            raise ValueError("Both dyn_fun and dyn_jac must be provided for generic propagation.")
        if not self.use_generic_propagation and self.n != 6:
            raise ValueError("Legacy propagation requires a 6-state vector. Provide dyn_fun/dyn_jac for other sizes.")

        if prop_settings is None:
            self.prop_settings = PropSettings(rtol=self.reltol, atol=self.abstol, method=self.method)
        else:
            self.prop_settings = prop_settings

        if not self.use_generic_propagation:
            if self.pos_idx != [0, 1, 2] or self.vel_idx != [3, 4, 5]:
                raise ValueError("Legacy 6-state mode assumes pos_idx=[0,1,2] and vel_idx=[3,4,5].")

    def propagate_state_and_stm_history(self, t_eval: np.ndarray):
        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
        if self.use_generic_propagation:
            X_hist, Phi_i0_hist = propagate_x_phi_history(
                x0=self.X0_star,
                t_eval=t_eval,
                f=self.dyn_fun,
                A=self.dyn_jac,
                settings=self.prop_settings,
            )
            return X_hist, Phi_i0_hist

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
        Phi_i0_hist = Y[:, 9:].reshape(-1, nx, nx)
        return Xstar_hist, Phi_i0_hist

    @staticmethod
    def phi_i0_to_phi_step(Phi_i0_hist: np.ndarray):
        Phi_i0_hist = np.asarray(Phi_i0_hist, dtype=float)
        N, n, _ = Phi_i0_hist.shape
        Phi_step = np.zeros_like(Phi_i0_hist)
        Phi_step[0] = np.eye(n)

        I = np.eye(n)
        for i in range(1, N):
            Phi_step[i] = Phi_i0_hist[i] @ np.linalg.solve(Phi_i0_hist[i - 1], I)

        return Phi_step

    def run(self, all_meas, get_measurement, predict_obs, H_matrix, Xtrue_meas: np.ndarray | None = None):
        if get_measurement is None or predict_obs is None or H_matrix is None:
            raise ValueError("get_measurement, predict_obs, and H_matrix must be provided for filter-agnostic run().")

        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))

        t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
        st_meas = [m["station"] for m in all_meas]
        N = len(all_meas)

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (N, self.n):
                raise ValueError(f"Xtrue_meas must have shape ({N}, {self.n}) aligned with sorted measurement order.")

        Xstar_hist, Phi_i0_hist = self.propagate_state_and_stm_history(t_meas)
        Phi_step = self.phi_i0_to_phi_step(Phi_i0_hist)

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
        I_m = np.eye(self.R.shape[0])

        for j in range(N):
            t = float(t_meas[j])
            meas_rec = all_meas[j]
            meas = get_measurement(meas_rec)
            have_meas = (meas is not None) and np.isfinite(meas).all()

            Xstar = Xstar_hist[j, :]
            Phi = Phi_step[j, :, :]

            xbar = Phi @ x_hat


            #ADdding State Noise cov stuff
            dt = 0.0 if j == 0 else float(t_meas[j] - t_meas[j - 1])
            snc_eci = self._snc_covariance(dt)
            Pbar = Phi @ P @ Phi.T + snc_eci

            C = predict_obs(Xstar, meas_rec) if have_meas else None

            if (C is not None) and np.isfinite(C).all():
                OminusC = meas - C
                residuals[j, :] = OminusC

                Htilde = H_matrix(Xstar, meas_rec)

                S = Htilde @ Pbar @ Htilde.T + self.R
                K = Pbar @ Htilde.T @ np.linalg.solve(S, I_m)

                x_hat = xbar + K @ (OminusC - Htilde @ xbar)

                A = I_n - K @ Htilde
                P = A @ Pbar @ A.T + K @ self.R @ K.T

                resid_pf[j, :] = OminusC - (Htilde @ x_hat)
            else:
                x_hat = xbar
                P = Pbar

            X_post = Xstar + x_hat
            X_pf[j, :] = X_post
            P_meas[j, :, :] = P
            P_pf[j, :] = P.reshape(-1, order="F")
            two_sigma[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P), 0.0))

            if have_meas:
                C_post = predict_obs(X_post, meas_rec)
                if C_post is not None and np.isfinite(C_post).all():
                    postfit_nl[j, :] = meas - C_post

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

        rms_final = self.print_rms_summary(label="LKF", first_pass_gap_s=self.first_pass_gap_s)

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
        }


class ExtendedKalmanFilter(KalmanFilterBase):
    """
    EKF in "state" form:
      - Propagate the current estimate Xhat with nonlinear dynamics from t_{i-1} -> t_i
      - Propagate covariance with STM Phi(t_i, t_{i-1}) from the same integration
      - Linearize measurement about Xbar (predicted state), update Xhat = Xbar + K*y
    """

    def __init__(
        self,
        x0: np.ndarray,
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
        state_mapping_dict: dict | None = None,
        dyn_fun=None,
        dyn_jac=None,
        prop_settings: PropSettings | None = None,
    ):
        super().__init__(x0, P0, R, Q, state_mapping_dict=state_mapping_dict)

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

        self.dyn_fun = dyn_fun
        self.dyn_jac = dyn_jac
        self.use_generic_propagation = (dyn_fun is not None) or (dyn_jac is not None)
        if self.use_generic_propagation and (dyn_fun is None or dyn_jac is None):
            raise ValueError("Both dyn_fun and dyn_jac must be provided for generic propagation.")
        if not self.use_generic_propagation and self.n != 6:
            raise ValueError("Legacy propagation requires a 6-state vector. Provide dyn_fun/dyn_jac for other sizes.")

        if prop_settings is None:
            self.prop_settings = PropSettings(rtol=self.reltol, atol=self.abstol, method=self.method)
        else:
            self.prop_settings = prop_settings

        if not self.use_generic_propagation:
            if self.pos_idx != [0, 1, 2] or self.vel_idx != [3, 4, 5]:
                raise ValueError("Legacy 6-state mode assumes pos_idx=[0,1,2] and vel_idx=[3,4,5].")

        self.Xhat = np.asarray(self.Xhat, dtype=float).reshape(self.n,)
        self.Phat = np.asarray(self.Phat, dtype=float).reshape(self.n, self.n)

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

    def _propagate_state_and_stm_step(self, t0: float, t1: float, X0: np.ndarray):
        t0 = float(t0)
        t1 = float(t1)
        X0 = np.asarray(X0, dtype=float).reshape(-1)

        if np.isclose(t1, t0):
            return X0.copy(), np.eye(self.n)

        if self.use_generic_propagation:
            X1, Phi_10 = propagate_x_phi_step(
                x0=X0,
                t0=t0,
                t1=t1,
                f=self.dyn_fun,
                A=self.dyn_jac,
                settings=self.prop_settings,
            )
            return X1, Phi_10

        nx = 6
        remove = np.array([6, 7, 8], dtype=int)
        X0_6 = X0.reshape(6,)
        X0_9 = np.hstack((X0_6, self.mu, self.J2, self.J3))

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
            (t0, t1),
            y0,
            t_eval=[t1],
            rtol=self.reltol,
            atol=self.abstol,
            method=self.method,
        )
        if not sol.success:
            raise RuntimeError(f"EKF STM integration failed: {sol.message}")

        yT = sol.y[:, -1]
        X1_6 = yT[:6].copy()
        Phi_10 = yT[9:].reshape(nx, nx).copy()
        return X1_6, Phi_10

    def run(self, all_meas, get_measurement, predict_obs, H_matrix, Xtrue_meas=None, t_prev_init=None):
        if get_measurement is None or predict_obs is None or H_matrix is None:
            raise ValueError("get_measurement, predict_obs, and H_matrix must be provided for filter-agnostic run().")

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
        I_m = np.eye(self.R.shape[0])

        for j in range(N):
            t = float(t_meas[j])
            meas_rec = all_meas[j]
            meas = get_measurement(meas_rec)
            have_meas = (meas is not None) and np.isfinite(meas).all()

            Xbar, Phi = self._propagate_state_and_stm_step(prev_time, t, X_hat)
            dt = float(t - prev_time)
            snc_eci = self._snc_covariance(dt)
            Pbar = Phi @ P @ Phi.T + snc_eci

            C = predict_obs(Xbar, meas_rec) if have_meas else None

            if (C is not None) and np.isfinite(C).all():
                OminusC = meas - C
                residuals[j, :] = OminusC

                Htilde = H_matrix(Xbar, meas_rec)

                S = Htilde @ Pbar @ Htilde.T + self.R
                K = Pbar @ Htilde.T @ np.linalg.solve(S, I_m)

                X_hat = Xbar + K @ OminusC
                A = I_n - K @ Htilde
                P = A @ Pbar @ A.T + K @ self.R @ K.T

                dx = X_hat - Xbar
                resid_pf[j, :] = OminusC - (Htilde @ dx)

                C_hat = predict_obs(X_hat, meas_rec)
                if C_hat is not None and np.isfinite(C_hat).all():
                    postfit_nl[j, :] = meas - C_hat
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

        rms_final = self.print_rms_summary(label="EKF", first_pass_gap_s=self.first_pass_gap_s)

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
        }

    def run_warmstarted(
        self,
        all_meas,
        lkf: "LinearizedKalmanFilter",
        get_measurement,
        predict_obs,
        H_matrix,
        num_init_meas: int = 100,
        Xtrue_meas: np.ndarray | None = None,
    ):
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
            return self.run(
                all_meas_sorted,
                get_measurement,
                predict_obs,
                H_matrix,
                Xtrue_meas=Xtrue_meas,
                t_prev_init=None,
            )

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

        lkf_out = lkf.run(init_meas, get_measurement, predict_obs, H_matrix, Xtrue_meas=Xtrue_init)

        X_start = np.asarray(lkf_out["Xhat_meas"][-1], dtype=float).reshape(self.n,)
        P_hist = lkf_out.get("P_meas", lkf_out.get("Phat_meas", None))
        if P_hist is None:
            raise KeyError("LKF output missing P_meas/Phat_meas needed for warmstart.")
        P_start = np.asarray(P_hist[-1], dtype=float).reshape(self.n, self.n)
        t_start = float(lkf_out["t_meas"][-1])

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

        ekf_out = self.run(
            rest_meas,
            get_measurement,
            predict_obs,
            H_matrix,
            Xtrue_meas=Xtrue_rest,
            t_prev_init=t_start,
        )

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
        if Ppf_lkf is None or Ppf_ekf is None:
            P_pf_comb = None
        else:
            P_pf_comb = np.vstack([Ppf_lkf, Ppf_ekf])

        if Xtrue_meas is None:
            err_comb = None
        else:
            err_comb = np.vstack([lkf_out["state_error_meas"], ekf_out["state_error_meas"]])

        rms_combined = KalmanFilterBase.compute_rms_summary_from_arrays(
            t=t_comb,
            postfit=postfit_nl_comb,
            state_err=err_comb,
            first_pass_gap_s=self.first_pass_gap_s,
            state_mapping_dict=self.state_mapping_dict,
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
        }
