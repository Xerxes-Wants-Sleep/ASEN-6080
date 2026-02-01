import numpy as np
from scipy.integrate import solve_ivp
from .jacobians import stm
from .range_rangerate import H_range_rangerate
 


  

class KalmanFilterBase:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, R: np.ndarray, Q: np.ndarray):
        # full estimated state (for convenience)
        self.Xhat = np.array(x0, dtype=float).copy()     # (6,)
        self.Phat = np.array(P0, dtype=float).copy()     # (6,6)
        self.R = np.array(R, dtype=float).copy()         # (2,2)
        self.Q = np.array(Q, dtype=float).copy()         # (6,6) discrete
        self.P0 = P0

        self.hist = {
            "t": [],
            "state_err": [],   # Xhat - Xtrue (if provided)
            "postfit": [],     # Y - G(Xhat,t) (NaNs allowed)
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
        # For "ignore first measurement" behavior:
        t = np.asarray(t, dtype=float).reshape(-1)

        keep_all = np.ones_like(t, dtype=bool)
        keep_ignore_first = np.ones_like(t, dtype=bool)
        if t.size > 0:
            keep_ignore_first[0] = False   # ignore first measurement only

        return keep_all, keep_ignore_first


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
    def compute_rms_summary_from_arrays(t, postfit, state_err=None, first_pass_gap_s=6*3600.0):
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
        }
    def reset_history(self):
        self.hist = {
            "t": [],
            "state_err": [],
            "postfit": [],
        }






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
        recenter_reference: bool = False,  # keep as an option, but default False
    ):
        super().__init__(X0_star, P0, R, Q)

        self.X0_star = np.asarray(X0_star, dtype=float).reshape(6,).copy()
        self.xhat = np.zeros(6, dtype=float)  # estimated error state
        self.xbar = np.zeros(6, dtype=float)  # predicted error state

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

    
    # Prop of ref
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
        Phi_i0_hist = Y[:, 9:].reshape(-1, nx, nx)  # Phi(t_i, t0) with a more simple inverse scheme apparently


        return Xstar_hist, Phi_i0_hist

    @staticmethod
    def phi_i0_to_phi_step(Phi_i0_hist: np.ndarray):
        """
        Convert Phi(t_i,t0) to Phi(t_i,t_{i-1}) via:
            Phi_i_im1 = Phi_i0 @ inv(Phi_im10)
                      = Phi_i0 @ solve(Phi_im10, I)
        """
        Phi_i0_hist = np.asarray(Phi_i0_hist, dtype=float)
        N = Phi_i0_hist.shape[0]
        Phi_step = np.zeros_like(Phi_i0_hist)
        Phi_step[0] = np.eye(6)

        I = np.eye(6)
        for i in range(1, N):
            # more stable than explicit inverse:
            Phi_step[i] = Phi_i0_hist[i] @ np.linalg.solve(Phi_i0_hist[i - 1], I)

        return Phi_step

    @staticmethod
    def G(station, X6: np.ndarray, t: float):
        d = station.measure(X6[:3], X6[3:], float(t))
        if d is None:
            return None
        return np.array([d["rho_km"], d["rho_dot_km_s"]], dtype=float)

    # ------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------
    def run(self, all_meas, stations, Xtrue_meas: np.ndarray | None = None):
        station_map = {st.name: st for st in stations}

        # Sort measurements by time
        all_meas_sorted = sorted(all_meas, key=lambda m: float(m["t"]))
        t_meas = np.array([float(m["t"]) for m in all_meas_sorted], dtype=float)
        st_meas = [m["station"] for m in all_meas_sorted]
        mcount = len(all_meas_sorted)

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (mcount, 6):
                raise ValueError(f"Xtrue_meas must have shape ({mcount}, 6) aligned with sorted measurement order.")

        # Precompute reference and Phi(t_i,t0), then build Phi(t_i,t_{i-1})
        Xstar_hist, Phi_i0_hist = self.propagate_state_and_stm_history(t_meas)
        Phi_step = self.phi_i0_to_phi_step(Phi_i0_hist)

        # outputs
        Xhat_hist = np.zeros((mcount, 6), dtype=float)
        postfit_resids = np.full((mcount, 2), np.nan, dtype=float)
        two_sigma = np.full((mcount, 6), np.nan, dtype=float)
        state_error = None if Xtrue_meas is None else np.zeros((mcount, 6), dtype=float)

        # init
        self.xhat[:] = 0.0
        self.Phat = np.array(self.P0, dtype = float).copy()
        self.Xhat = self.X0_star + self.xhat
        Phat_hist = np.zeros((mcount, 6, 6), dtype=float)


        for i, m in enumerate(all_meas_sorted):
            t_i = float(m["t"])
            st = station_map[m["station"]]

            Xstar_i = Xstar_hist[i, :]
            Phi_i_im1 = Phi_step[i, :, :]

            # TIME UPDATE
            self.xbar = Phi_i_im1 @ self.xhat
            Pbar_i = Phi_i_im1 @ self.Phat @ Phi_i_im1.T + self.Q

            # measurement vector
            Y_i = np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)

            # If station not visible / masked measurement, skip update
            Gstar_i = self.G(st, Xstar_i, t_i)
            if Gstar_i is None:
                self.xhat = self.xbar
                self.Phat = Pbar_i
                Phat_hist[i, :, :] = self.Phat
                self.Xhat = Xstar_i + self.xhat

                Xhat_hist[i, :] = self.Xhat
                two_sigma[i, :] = 2.0 * np.sqrt(np.maximum(np.diag(self.Phat), 0.0))
                postfit_resids[i, :] = np.array([np.nan, np.nan], dtype=float)

                self.log_epoch(t_i, postfit_resid=postfit_resids[i, :],
                               Xtrue=(None if Xtrue_meas is None else Xtrue_meas[i, :]))
            else:
                # prefit residual at reference (used internally)
                y_i = Y_i - Gstar_i

                # measurement partials at reference
                v_zero = np.zeros(3)
                Rs, Vs, _ = st.ecef2eci(t_i, st.r_ecef, v_zero)
                Htilde_i = H_range_rangerate(Xstar_i[:3], Xstar_i[3:], Rs, Vs)

                # gain
                S_i = Htilde_i @ Pbar_i @ Htilde_i.T + self.R
                K_i = Pbar_i @ Htilde_i.T @ np.linalg.solve(S_i, np.eye(S_i.shape[0]))

                # update
                self.xhat = self.xbar + K_i @ (y_i - Htilde_i @ self.xbar)

                I6 = np.eye(6)
                self.Phat = (I6 - K_i @ Htilde_i) @ Pbar_i @ (I6 - K_i @ Htilde_i).T + K_i @ self.R @ K_i.T
                Phat_hist[i, :, :] = self.Phat
                self.Xhat = Xstar_i + self.xhat
                Xhat_hist[i, :] = self.Xhat
                two_sigma[i, :] = 2.0 * np.sqrt(np.maximum(np.diag(self.Phat), 0.0))

                # NONLINEAR post-fit residual for HW2:
                #   r_i^+ = Y_i - G(Xhat_i, t_i)
                Ghat_i = self.G(st, self.Xhat, t_i)
                r_post = (Y_i - Ghat_i) if (Ghat_i is not None) else np.array([np.nan, np.nan], dtype=float)
                postfit_resids[i, :] = r_post

                self.log_epoch(t_i, postfit_resid=r_post,
                               Xtrue=(None if Xtrue_meas is None else Xtrue_meas[i, :]))

            if state_error is not None:
                state_error[i, :] = self.Xhat - Xtrue_meas[i, :]


        rms_final = self.print_rms_summary(label="LKF", first_pass_gap_s=self.first_pass_gap_s)

        return {
            "t_meas": t_meas,
            "station_meas": st_meas,
            "Xhat_meas": Xhat_hist,
            "state_error_meas": state_error,
            "postfit_resids_meas": postfit_resids,
            "two_sigma_meas": two_sigma,
            "rms_final": rms_final,
            "Phat_meas": Phat_hist
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
    ):
        super().__init__(x0, P0, R, Q)

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

        # Make sure base state is 6-vector
        self.Xhat = np.asarray(self.Xhat, dtype=float).reshape(6,)
        self.Phat = np.asarray(self.Phat, dtype=float).reshape(6, 6)


    @staticmethod
    def print_rms_summary_dict(rms: dict, label: str = ""):
        hdr = f"\n===== RMS SUMMARY {label} =====" if label else "\n===== RMS SUMMARY ====="
        print(hdr)

        if rms.get("state_comp_all") is not None:
            print("State error RMS (component-wise) [all]:")
            print(rms["state_comp_all"])
            print("State error RMS (component-wise) [ignore first]:")
            print(rms["state_comp_ignore_first"])
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



    # measurement model
    @staticmethod
    def G(station, X6: np.ndarray, t: float):
        d = station.measure(X6[:3], X6[3:], float(t))
        if d is None:
            return None
        return np.array([d["rho_km"], d["rho_dot_km_s"]], dtype=float)

    def _propagate_state_and_stm_step(self, t0: float, t1: float, X0_6: np.ndarray):
        """
        Integrate nonlinear state and 6x6 STM from t0->t1 with initial STM=I.
        """
        t0 = float(t0)
        t1 = float(t1)

        # Handle zero-length step
        if np.isclose(t1, t0):
            return X0_6.copy(), np.eye(6)

        nx = 6
        remove = np.array([6, 7, 8], dtype=int)  # remove mu,J2,J3 from Jacobian/STM

        X0_6 = np.asarray(X0_6, dtype=float).reshape(6,)
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
        Phi_10 = yT[9:].reshape(nx, nx).copy()  # Phi(t1,t0)

        return X1_6, Phi_10

    def run(self, all_meas, stations, Xtrue_meas: np.ndarray | None = None, t_prev_init: float | None = None):
        station_map = {st.name: st for st in stations}

        # Sort measurements by time
        all_meas_sorted = sorted(all_meas, key=lambda m: float(m["t"]))
        t_meas = np.array([float(m["t"]) for m in all_meas_sorted], dtype=float)
        st_meas = [m["station"] for m in all_meas_sorted]
        mcount = len(all_meas_sorted)

        if t_prev_init is None:
            t_prev = float(t_meas[0])
        else:
            t_prev = float(t_prev_init)

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (mcount, 6):
                raise ValueError(f"Xtrue_meas must have shape ({mcount}, 6) aligned with sorted measurement order.")

        # outputs
        Xhat_hist = np.zeros((mcount, 6), dtype=float)
        postfit_resids = np.full((mcount, 2), np.nan, dtype=float)
        two_sigma = np.full((mcount, 6), np.nan, dtype=float)
        state_error = None if Xtrue_meas is None else np.zeros((mcount, 6), dtype=float)

        # init
        self.Xhat = np.asarray(self.Xhat, dtype=float).reshape(6,)
        self.Phat = np.asarray(self.Phat, dtype=float).reshape(6, 6)



        for i, m in enumerate(all_meas_sorted):
            t_i = float(m["t"])
            st = station_map[m["station"]]

            # 1) TIME UPDATE
            #  (propagate nonlinear state + STM)
            #  Use current post state as initial condition
            Xbar_i, Phi_i_im1 = self._propagate_state_and_stm_step(t_prev, t_i, self.Xhat)

            Pbar_i = Phi_i_im1 @ self.Phat @ Phi_i_im1.T + self.Q

            # measurement vector
            Y_i = np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)

            # If station not visible / masked measurement, skip update
            Gbar = self.G(st, Xbar_i, t_i)
            if Gbar is None:
                self.Xhat = Xbar_i
                self.Phat = Pbar_i

                Xhat_hist[i, :] = self.Xhat
                two_sigma[i, :] = 2.0 * np.sqrt(np.maximum(np.diag(self.Phat), 0.0))
                postfit_resids[i, :] = np.array([np.nan, np.nan], dtype=float)

                self.log_epoch(
                    t_i,
                    postfit_resid=postfit_resids[i, :],
                    Xtrue=(None if Xtrue_meas is None else Xtrue_meas[i, :]),
                )
            else:
                # 2) MEASUREMENT UPDATE 
                y_i = Y_i - Gbar  # innovation about predicted state

                v_zero = np.zeros(3)
                Rs, Vs, _ = st.ecef2eci(t_i, st.r_ecef, v_zero)
                Htilde_i = H_range_rangerate(Xbar_i[:3], Xbar_i[3:], Rs, Vs)

                S_i = Htilde_i @ Pbar_i @ Htilde_i.T + self.R

                # K = Pbar H^T S^{-1} (fancy stable inverse)
                K_i = Pbar_i @ Htilde_i.T @ np.linalg.solve(S_i, np.eye(S_i.shape[0]))

                # state update
                self.Xhat = Xbar_i + K_i @ y_i

                # covariance update
                I6 = np.eye(6)
                A = (I6 - K_i @ Htilde_i)
                self.Phat = A @ Pbar_i @ A.T + K_i @ self.R @ K_i.T

                Xhat_hist[i, :] = self.Xhat
                two_sigma[i, :] = 2.0 * np.sqrt(np.maximum(np.diag(self.Phat), 0.0))

                # NONLINEAR post-fit residual
                Ghat_i = self.G(st, self.Xhat, t_i)
                r_post = (Y_i - Ghat_i) if (Ghat_i is not None) else np.array([np.nan, np.nan], dtype=float)
                postfit_resids[i, :] = r_post

                self.log_epoch(
                    t_i,
                    postfit_resid=r_post,
                    Xtrue=(None if Xtrue_meas is None else Xtrue_meas[i, :]),
                )

            if state_error is not None:
                state_error[i, :] = self.Xhat - Xtrue_meas[i, :]

            # advance
            t_prev = t_i

        rms_final = self.print_rms_summary(label="EKF", first_pass_gap_s=self.first_pass_gap_s)

        return {
            "t_meas": t_meas,
            "station_meas": st_meas,
            "Xhat_meas": Xhat_hist,
            "state_error_meas": state_error,
            "postfit_resids_meas": postfit_resids,
            "two_sigma_meas": two_sigma,
            "rms_final": rms_final,
        }
    
    def run_warmstarted(
        self,
        all_meas,
        stations,
        lkf: "LinearizedKalmanFilter",
        num_init_meas: int = 100,
        Xtrue_meas: np.ndarray | None = None,
    ):
        """
        Warm-start EKF using an LKF initialization on the first num_init_meas measurements.

        Steps:
          1) Sort measurements by time
          2) Run LKF on first N meas
          3) Set EKF initial (Xhat, Phat) = last LKF estimate
          4) Run EKF on remaining meas, using t_prev_init = last LKF time
          5) Return combined history + both sub-outputs
        """

        # ---- sort once here so LKF and EKF split consistently ----
        all_meas_sorted = sorted(all_meas, key=lambda m: float(m["t"]))
        mcount = len(all_meas_sorted)

        if mcount == 0:
            return {
                "lkf_init": None,
                "ekf": None,
                "t_meas": np.array([], dtype=float),
                "station_meas": [],
                "Xhat_meas": np.empty((0, 6), dtype=float),
                "state_error_meas": None,
                "postfit_resids_meas": np.empty((0, 2), dtype=float),
                "two_sigma_meas": np.empty((0, 6), dtype=float),
                "rms_combined": None,
            }

        N = int(num_init_meas)
        if N <= 0:
            # no warmstart; just run EKF normally
            return self.run(all_meas_sorted, stations, Xtrue_meas=Xtrue_meas, t_prev_init=None)

        N = min(N, mcount)
        init_meas = all_meas_sorted[:N]
        rest_meas = all_meas_sorted[N:]

        # Truth slicing: assumes Xtrue_meas is ALREADY aligned with sorted measurement order
        Xtrue_init = None
        Xtrue_rest = None
        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (mcount, 6):
                raise ValueError(f"Xtrue_meas must have shape ({mcount}, 6) aligned with sorted measurement order.")
            Xtrue_init = Xtrue_meas[:N, :]
            Xtrue_rest = Xtrue_meas[N:, :]

        # ---- reset histories so repeated runs don't append ----
        if hasattr(lkf, "reset_history"):
            lkf.reset_history()
        self.reset_history()

        # ---- 1) Run LKF init ----
        lkf_out = lkf.run(init_meas, stations, Xtrue_meas=Xtrue_init)

        # Pull last LKF posterior as EKF start
        X_start = np.asarray(lkf_out["Xhat_meas"][-1], dtype=float).reshape(6,)
        P_start = np.asarray(lkf_out["Phat_meas"][-1], dtype=float).reshape(6, 6)
        t_start = float(lkf_out["t_meas"][-1])

        # ---- 2) Initialize EKF with LKF posterior ----
        self.Xhat = X_start.copy()
        self.Phat = P_start.copy()

        # If there's nothing left, just return LKF as the "combined"
        if len(rest_meas) == 0:
            t_comb = np.asarray(lkf_out["t_meas"], dtype=float)
            post_comb = np.asarray(lkf_out["postfit_resids_meas"], dtype=float)
            err_comb = None if lkf_out["state_error_meas"] is None else np.asarray(lkf_out["state_error_meas"], dtype=float)

            rms_combined = KalmanFilterBase.compute_rms_summary_from_arrays(
                t=t_comb,
                postfit=post_comb,
                state_err=err_comb,
                first_pass_gap_s=self.first_pass_gap_s,
            )

            return {
                "lkf_init": lkf_out,
                "ekf": None,
                "t_meas": t_comb,
                "station_meas": lkf_out["station_meas"],
                "Xhat_meas": lkf_out["Xhat_meas"],
                "state_error_meas": lkf_out["state_error_meas"],
                "postfit_resids_meas": lkf_out["postfit_resids_meas"],
                "two_sigma_meas": lkf_out["two_sigma_meas"],
                "rms_combined": rms_combined,
                "t_start_ekf": t_start,
            }

        # ---- 3) Run EKF from where LKF left off ----
        ekf_out = self.run(rest_meas, stations, Xtrue_meas=Xtrue_rest, t_prev_init=t_start)

        # ---- 4) Stitch outputs ----
        t_comb = np.hstack([lkf_out["t_meas"], ekf_out["t_meas"]])
        station_comb = lkf_out["station_meas"] + ekf_out["station_meas"]
        Xhat_comb = np.vstack([lkf_out["Xhat_meas"], ekf_out["Xhat_meas"]])
        post_comb = np.vstack([lkf_out["postfit_resids_meas"], ekf_out["postfit_resids_meas"]])
        two_sigma_comb = np.vstack([lkf_out["two_sigma_meas"], ekf_out["two_sigma_meas"]])

        if Xtrue_meas is None:
            err_comb = None
        else:
            err_comb = np.vstack([lkf_out["state_error_meas"], ekf_out["state_error_meas"]])

        rms_combined = KalmanFilterBase.compute_rms_summary_from_arrays(
            t=t_comb,
            postfit=post_comb,
            state_err=err_comb,
            first_pass_gap_s=self.first_pass_gap_s,
        )

        return {
            "lkf_init": lkf_out,
            "ekf": ekf_out,
            "t_start_ekf": t_start,
            "t_meas": t_comb,
            "station_meas": station_comb,
            "Xhat_meas": Xhat_comb,
            "state_error_meas": err_comb,
            "postfit_resids_meas": post_comb,
            "two_sigma_meas": two_sigma_comb,
            "rms_combined": rms_combined,
        }