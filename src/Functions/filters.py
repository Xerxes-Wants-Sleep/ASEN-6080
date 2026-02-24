import numpy as np
from scipy.integrate import solve_ivp
from .jacobians import stm
from .range_rangerate import H_range_rangerate
from .snc import state_noise_compensation, eci_to_ric_rotation
 


  

class KalmanFilterBase:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, R: np.ndarray, Q: np.ndarray):
        # full estimated state (for convenience)
        self.Xhat = np.array(x0, dtype=float).copy()     # (6,)
        self.Phat = np.array(P0, dtype=float).copy()     # (6,6)
        self.R = np.array(R, dtype=float).copy()         # (2,2)
        self.Q = np.array(Q, dtype=float).copy()         # (3,3) continuous accel or (6,6) discrete
        if self.Q.shape not in {(3, 3), (6, 6)}:
            raise ValueError(
                f"Q must be shape (3,3) for SNC accel covariance or (6,6) for fixed discrete noise; got {self.Q.shape}"
            )
        # Process-noise frame for 3x3 acceleration covariance:
        #   "eci" -> Q already in ECI
        #   "ric" -> Q defined in RIC and rotated to ECI each step
        self.q_frame = "eci"
        self.P0 = P0

        self.hist = {
            "t": [],
            "state_err": [],   # Xhat - Xtrue (if provided)
            "postfit": [],     # Y - G(Xhat,t) (NaNs allowed)
        }

    def build_process_noise(self, delta_t: float, r_eci: np.ndarray | None = None, v_eci: np.ndarray | None = None) -> np.ndarray:
        """
        Return discrete Q_k for the current propagation step.

        - If self.Q is (3,3), treat it as continuous acceleration covariance and
          build Q_k with state-noise compensation.
        - If self.Q is (6,6), treat it as already-discrete fixed Q_k.
        """
        if self.Q.shape == (6, 6):
            return self.Q

        dt = max(float(delta_t), 0.0)
        if dt == 0.0:
            return np.zeros((6, 6), dtype=float)
        
        

        q_frame = str(getattr(self, "q_frame", "eci")).strip().lower()
        if q_frame == "eci":
            Q_accel_eci = self.Q
        elif q_frame == "ric":
            if r_eci is None or v_eci is None:
                raise ValueError("RIC process noise requested but r_eci/v_eci were not provided.")
            C_eci_to_ric = eci_to_ric_rotation(r_eci=r_eci, v_eci=v_eci)
            C_ric_to_eci = C_eci_to_ric.T
            Q_accel_eci = C_ric_to_eci @ self.Q @ C_ric_to_eci.T
        else:
            raise ValueError(f"Unsupported q_frame '{self.q_frame}'. Use 'eci' or 'ric'.")

        return state_noise_compensation(delta_t=dt, n=6, m=3, Q=Q_accel_eci)

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
        rho = d["rho_km"] if "rho_km" in d else d["rho"]
        rhod = d["rho_dot_km_s"] if "rho_dot_km_s" in d else d["rho_dot"]
        return np.array([rho, rhod], dtype=float)
    





    @staticmethod
    def rts_smoother(
        x_filt_hist,
        P_filt_hist,
        x_pred_hist,
        P_pred_hist,
        Phi_step_hist,
        smooth_back_points: int | None = None,
    ):
        """
        Rauch–Tung–Striebel smoother for the linear time-varying system.

        Inputs:
        x_filt_hist[k] = x_{k|k}
        P_filt_hist[k] = P_{k|k}
        x_pred_hist[k] = x_{k|k-1}
        P_pred_hist[k] = P_{k|k-1}
        Phi_step_hist[k] = Phi_{k,k-1}   (with Phi_step_hist[0] = I)

        smooth_back_points:
        None  -> smooth full interval (fixed-interval RTS)
        K     -> smooth only the last K points backward (earlier points remain filtered)

        Returns:
        x_smooth[k] = x_{k|N}
        P_smooth[k] = P_{k|N}
        """
        x_filt_hist = np.asarray(x_filt_hist, dtype=float)
        P_filt_hist = np.asarray(P_filt_hist, dtype=float)
        x_pred_hist = np.asarray(x_pred_hist, dtype=float)
        P_pred_hist = np.asarray(P_pred_hist, dtype=float)
        Phi_step_hist = np.asarray(Phi_step_hist, dtype=float)

        N = x_filt_hist.shape[0]
        x_smooth = x_filt_hist.copy()
        P_smooth = P_filt_hist.copy()
        if N == 0:
            return x_smooth, P_smooth

        I6 = np.eye(6)

        # initialize at final time
        # x_smooth[N-1] = x_{N-1|N-1}
        # P_smooth[N-1] = P_{N-1|N-1}

        if smooth_back_points is None:
            k_min = 0
        else:
            Kback = int(smooth_back_points)
            if Kback <= 1:
                return x_smooth, P_smooth
            Kback = min(Kback, N)
            k_min = N - Kback

        for k in range(N - 2, k_min - 1, -1):
            Phi_kp1_k = Phi_step_hist[k + 1]      # Phi_{k+1,k}
            P_k_k     = P_filt_hist[k]
            P_kp1_k   = P_pred_hist[k + 1]        # P_{k+1|k}

            # Ck = P_k_k Phi^T (P_{k+1|k})^{-1}  (use solve, not inv)
            # Solve: P_kp1_k * X = I  -> X = inv(P_kp1_k)
            invP = np.linalg.solve(P_kp1_k, I6)
            Ck = (P_k_k @ Phi_kp1_k.T) @ invP

            x_smooth[k] = x_filt_hist[k] + Ck @ (x_smooth[k + 1] - x_pred_hist[k + 1])
            P_smooth[k] = P_filt_hist[k] + Ck @ (P_smooth[k + 1] - P_pred_hist[k + 1]) @ Ck.T

            # optional: enforce symmetry (helps numerics)
            P_smooth[k] = 0.5 * (P_smooth[k] + P_smooth[k].T)

        return x_smooth, P_smooth

    # ------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------
    def run(
        self,
        all_meas,
        stations,
        Xtrue_meas: np.ndarray | None = None,
        run_smoother: bool = False,
        smooth_back_points: int | None = None,
    ):
        # -----------------------------
        # 0) Setup / sort
        # -----------------------------
        station_map = {st.name: st for st in stations}
        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))

        t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
        st_meas = [m["station"] for m in all_meas]
        N = len(all_meas)

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (N, 6):
                raise ValueError(f"Xtrue_meas must have shape ({N}, 6) aligned with sorted measurement order.")

        # -----------------------------
        # 1) Precompute reference + Phi steps
        # -----------------------------
        Xstar_hist, Phi_i0_hist = self.propagate_state_and_stm_history(t_meas)  # X*(t_i), Phi(t_i,t0)
        Phi_step = self.phi_i0_to_phi_step(Phi_i0_hist)                         # Phi(t_i,t_{i-1})

        # -----------------------------
        # 2) Allocate outputs
        # -----------------------------
        residuals = np.full((N, 2), np.nan, dtype=float)    # prefit: OminusC
        resid_pf  = np.full((N, 2), np.nan, dtype=float)    # linear postfit: OminusC - H*x_hat(+)
        postfit_nl = np.full((N, 2), np.nan, dtype=float)   # nonlinear postfit: Y - h(X_pf)

        X_pf  = np.full((N, 6), np.nan, dtype=float)        # post-fit state solution
        P_meas = np.full((N, 6, 6), np.nan, dtype=float)    # post-fit covariance (3D)
        P_pf  = np.full((N, 36), np.nan, dtype=float)       # Reshape(Covariance)
        two_sigma = np.full((N, 6), np.nan, dtype=float)
        x_pred_hist = np.zeros((N, 6), dtype=float)         # x_{k|k-1}
        P_pred_hist = np.zeros((N, 6, 6), dtype=float)      # P_{k|k-1}
        x_filt_hist = np.zeros((N, 6), dtype=float)         # x_{k|k}
        P_filt_hist = np.zeros((N, 6, 6), dtype=float)      # P_{k|k}
        # Phi_step already exists as (N,6,6) with Phi_step[j] = Phi_{j,j-1}

        state_error = None if Xtrue_meas is None else np.full((N, 6), np.nan, dtype=float)

        # -----------------------------
        # 3) Init filter vars (locals)
        # -----------------------------
        x_hat = np.zeros(6, dtype=float)                    # error-state estimate
        P = np.array(self.P0, dtype=float).copy()           # error-state covariance

        # -----------------------------
        # 4) Main filter loop
        # -----------------------------
        for j in range(N):

            t = float(t_meas[j])
            st = station_map[st_meas[j]]

            Y = np.array([all_meas[j]["rho_km"], all_meas[j]["rho_dot_km_s"]], dtype=float)
            Xstar = Xstar_hist[j, :]
            Phi = Phi_step[j, :, :]

            # ---- Time Update (error-state) ----
            dt = 0.0 if j == 0 else (t - float(t_meas[j - 1]))
            Qk = self.build_process_noise(dt, r_eci=Xstar[:3], v_eci=Xstar[3:6])
            xbar = Phi @ x_hat
            Pbar = Phi @ P @ Phi.T + Qk
            x_pred_hist[j, :] = xbar
            P_pred_hist[j, :, :] = Pbar

            # ---- Observation at reference ----
            # Compute station ECI once and reuse for G(), Htilde, and postfit
            Rs, Vs, _ = st.ecef2eci(t, st.r_ecef, np.zeros(3))
            C = self.G(st, Xstar, t)
            if C is not None:

                # Prefit residual 
                OminusC = Y - C
                residuals[j, :] = OminusC

                # Linearize measurement about reference X*
                Htilde = H_range_rangerate(Xstar[:3], Xstar[3:], Rs, Vs)  # (2,6)

                # Kalman gain
                S = Htilde @ Pbar @ Htilde.T + self.R
                K = Pbar @ Htilde.T @ np.linalg.solve(S, np.eye(2))

                # ---- Measurement Update
                x_hat = xbar + K @ (OminusC - Htilde @ xbar)

                A = np.eye(6) - K @ Htilde
                P = A @ Pbar @ A.T + K @ self.R @ K.T  # Joseph

                # Linear post-fit residual 
                resid_pf[j, :] = OminusC - (Htilde @ x_hat)

            else:
                # No measurement update
                x_hat = xbar
                P = Pbar

            x_filt_hist[j, :] = x_hat
            P_filt_hist[j, :, :] = P

            # ---- Post-fit state + store outputs ----
            X_post = Xstar + x_hat
            X_pf[j, :] = X_post
            P_meas[j, :, :] = P
            P_pf[j, :] = P.reshape(-1, order="F")  
            two_sigma[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P), 0.0))

            # Nonlinear post-fit residual (Rs/Vs already computed above, reused via measure())
            C_post = self.G(st, X_post, t)
            postfit_nl[j, :] = (Y - C_post) if (C_post is not None) else np.array([np.nan, np.nan], dtype=float)

            if state_error is not None:
                state_error[j, :] = X_post - Xtrue_meas[j, :]

            # Keep base-class history consistent with your RMS summary
            self.Xhat = X_post
            self.log_epoch(t, postfit_resid=postfit_nl[j, :],
                        Xtrue=(None if Xtrue_meas is None else Xtrue_meas[j, :]))

        # sync final state back to object
        self.xhat = x_hat
        self.Phat = P
        self.Xhat = X_pf[-1, :]

        rms_final = self.print_rms_summary(label="LKF", first_pass_gap_s=self.first_pass_gap_s)

        smoother_ran = bool(run_smoother)
        x_smooth_err = None
        X_smooth = None
        P_smooth = None
        three_sigma_smooth = None
        state_error_smooth = None
        postfit_lin_smooth = None

        if smoother_ran:
            x_smooth_err, P_smooth = self.rts_smoother(
                x_filt_hist=x_filt_hist,
                P_filt_hist=P_filt_hist,
                x_pred_hist=x_pred_hist,
                P_pred_hist=P_pred_hist,
                Phi_step_hist=Phi_step,
                smooth_back_points=smooth_back_points,
            )

            X_smooth = Xstar_hist + x_smooth_err
            three_sigma_smooth = 3.0 * np.sqrt(
                np.maximum(np.diagonal(P_smooth, axis1=1, axis2=2), 0.0)
            )

            if Xtrue_meas is not None:
                state_error_smooth = X_smooth - Xtrue_meas

            # Derived linearized post-fit residual using smoothed error-state:
            #   r_pf,lin,smooth = (O - C_ref) - H_tilde x_{k|N}
            postfit_lin_smooth = np.full((N, 2), np.nan, dtype=float)
            for j in range(N):
                t = float(t_meas[j])
                st = station_map[st_meas[j]]
                Xstar = Xstar_hist[j, :]

                C_ref = self.G(st, Xstar, t)
                if C_ref is None:
                    continue

                Rs, Vs, _ = st.ecef2eci(t, st.r_ecef, np.zeros(3))
                Htilde = H_range_rangerate(Xstar[:3], Xstar[3:], Rs, Vs)
                postfit_lin_smooth[j, :] = residuals[j, :] - (Htilde @ x_smooth_err[j, :])

        return {
            "t_meas": t_meas,
            "station_meas": st_meas,

            # states
            "xhat_meas": X_pf,
            "Xhat_meas": X_pf,    # use post-fit state history (what you plot)
            "X_pf": X_pf,

            # residuals 
            "prefit_resids_final": residuals,
            "postfit_resids_linear_final": resid_pf,
            "postfit_resids_meas": postfit_nl,

            # covariance
            "P_meas": P_meas,
            "Phat_meas": P_meas,  # alias (helps later if you warmstart)
            "P_pf": P_pf,
            "x_pred_hist": x_pred_hist,
            "P_pred_hist": P_pred_hist,
            "x_filt_hist": x_filt_hist,
            "P_filt_hist": P_filt_hist,
            "Phi_step": Phi_step,
            "smoother_ran": smoother_ran,
            "x_smooth_err": x_smooth_err,          # smoothed error-state wrt reference X*(t)
            "Xhat_smooth": X_smooth,               # smoothed full state = X*(t) + x_smooth_err
            "P_smooth": P_smooth,                  # smoothed covariance
            "two_sigma_smooth": three_sigma_smooth,  # actually +/-3 sigma (name kept for compatibility)
            "state_error_smooth_meas": state_error_smooth,
            "postfit_resids_linear_smooth": postfit_lin_smooth,

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
        rho = d["rho_km"] if "rho_km" in d else d["rho"]
        rhod = d["rho_dot_km_s"] if "rho_dot_km_s" in d else d["rho_dot"]
        return np.array([rho, rhod], dtype=float)

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

    def run(self, all_meas, stations, Xtrue_meas=None, t_prev_init=None):
        # -----------------------------
        # 0) Setup / sort / early exit
        # -----------------------------
        station_map = {st.name: st for st in stations}
        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
        N = len(all_meas)

        t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
        st_meas = [m["station"] for m in all_meas]

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (N, 6):
                raise ValueError(f"Xtrue_meas must have shape ({N}, 6) aligned with sorted measurement order.")

        # -----------------------------
        # 1) Allocate outputs
        # -----------------------------
        residuals = np.full((N, 2), np.nan, dtype=float)     # prefit (OminusC)
        resid_pf  = np.full((N, 2), np.nan, dtype=float)     # linear postfit
        postfit_nl = np.full((N, 2), np.nan, dtype=float)    # nonlinear postfit (keep)

        X_pf = np.full((N, 6), np.nan, dtype=float)          # post-fit state history
        P_meas = np.full((N, 6, 6), np.nan, dtype=float)     # post-fit covariance history
        P_pf = np.full((N, 36), np.nan, dtype=float)        
        two_sigma = np.full((N, 6), np.nan, dtype=float)

        state_error = None if Xtrue_meas is None else np.full((N, 6), np.nan, dtype=float)

        # -----------------------------
        # 2) Init filter vars (locals)
        # -----------------------------
        X_hat = np.asarray(self.Xhat, dtype=float).reshape(6,)
        P = np.asarray(self.Phat, dtype=float).reshape(6, 6)

        prev_time = float(t_meas[0]) if t_prev_init is None else float(t_prev_init)

        # -----------------------------
        # 3) Main filter loop
        # -----------------------------
        for j in range(N):

            t = float(t_meas[j])
            st = station_map[st_meas[j]]
            Y = np.array([all_meas[j]["rho_km"], all_meas[j]["rho_dot_km_s"]], dtype=float)

            # ---- Time Update (propagate state + STM) ----
            dt = t - prev_time
            Xbar, Phi = self._propagate_state_and_stm_step(prev_time, t, X_hat)
            Qk = self.build_process_noise(dt, r_eci=Xbar[:3], v_eci=Xbar[3:6])
            Pbar = Phi @ P @ Phi.T + Qk

            # ---- Observation / prefit residual ----
            # Compute station ECI once and reuse for G(), Htilde, and postfit
            Rs, Vs, _ = st.ecef2eci(t, st.r_ecef, np.zeros(3))
            C = self.G(st, Xbar, t)
            if C is not None:

                OminusC = Y - C
                residuals[j, :] = OminusC

                # ---- Linearize measurement ----
                Htilde = H_range_rangerate(Xbar[:3], Xbar[3:], Rs, Vs)

                # ---- Kalman gain ----
                S = Htilde @ Pbar @ Htilde.T + self.R
                K = Pbar @ Htilde.T @ np.linalg.solve(S, np.eye(2))

                # ---- Measurement Update ----
                X_hat = Xbar + K @ OminusC
                A = np.eye(6) - K @ Htilde
                P = A @ Pbar @ A.T + K @ self.R @ K.T

                # ---- Linear post-fit residual
                x_hat_err = X_hat - Xbar
                resid_pf[j, :] = OminusC - (Htilde @ x_hat_err)

                # ---- Nonlinear post-fit residual (reuse Rs, Vs already computed) ----
                C_hat = self.G(st, X_hat, t)
                postfit_nl[j, :] = (Y - C_hat) if (C_hat is not None) else np.array([np.nan, np.nan], dtype=float)


            else:
                # No measurement update
                X_hat, P = Xbar, Pbar

            # ---- Store outputs (single block) ----
            X_pf[j, :] = X_hat
            P_meas[j, :, :] = P
            P_pf[j, :] = P.reshape(-1, order="F") 
            two_sigma[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P), 0.0))

            if state_error is not None:
                state_error[j, :] = X_hat - Xtrue_meas[j, :]

            # advance time
            prev_time = t

            # log nonlinear postfit into base history (optional)
            self.Xhat = X_hat
            self.log_epoch(t, postfit_resid=postfit_nl[j, :],
                        Xtrue=(None if Xtrue_meas is None else Xtrue_meas[j, :]))

        # sync state back
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
            "rms_by_iter": None
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
        Warm-start EKF using LKF on the first num_init_meas observations.

        
        - run a filter to get a posterior (state + covariance) at some time
        - use that posterior as the next filter's initial condition
        - continue from that time forward

        Returns a SINGLE combined output dict with the same keys the batch
        plotting pipeline expects (so run_filter_post_processing() works).
        """

        # ---- sort once so the split is consistent ----
        all_meas_sorted = sorted(all_meas, key=lambda m: float(m["t"]))
        mcount = len(all_meas_sorted)

        if mcount == 0:
            return {
                "lkf_init": None,
                "ekf": None,
                "t_meas": np.array([], dtype=float),
                "station_meas": [],
                "Xhat_meas": np.empty((0, 6), dtype=float),
                "xhat_meas": np.empty((0, 6), dtype=float),
                "P_meas": np.empty((0, 6, 6), dtype=float),
                "two_sigma_meas": np.empty((0, 6), dtype=float),
                "state_error_meas": None,
                "prefit_resids_final": np.empty((0, 2), dtype=float),
                "postfit_resids_linear_final": np.empty((0, 2), dtype=float),
                "postfit_resids_meas": np.empty((0, 2), dtype=float),
                "P_pf": np.empty((0, 36), dtype=float),
                "rms_final": None,
                "rms_by_iter": None,
            }

        # Truth must be aligned to the *same sorted order*
        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (mcount, 6):
                raise ValueError(
                    f"Xtrue_meas must have shape ({mcount}, 6) aligned with sorted measurement order."
                )

        # ---- choose init length ----
        Ninit = int(num_init_meas)
        if Ninit <= 0:
            # no warmstart; just run EKF normally on all measurements
            return self.run(all_meas_sorted, stations, Xtrue_meas=Xtrue_meas, t_prev_init=None)

        Ninit = min(Ninit, mcount)

        init_meas = all_meas_sorted[:Ninit]
        rest_meas = all_meas_sorted[Ninit:]

        Xtrue_init = None
        Xtrue_rest = None
        if Xtrue_meas is not None:
            Xtrue_init = Xtrue_meas[:Ninit, :]
            Xtrue_rest = Xtrue_meas[Ninit:, :]

        # ---- reset histories so repeat calls don’t append ----
        if hasattr(lkf, "reset_history"):
            lkf.reset_history()
        if hasattr(self, "reset_history"):
            self.reset_history()

        # ------------------------------------------------------------
        # 1) Run LKF on first chunk
        # ------------------------------------------------------------
        lkf_out = lkf.run(init_meas, stations, Xtrue_meas=Xtrue_init)

        # Pull last LKF posterior state
        X_start = np.asarray(lkf_out["Xhat_meas"][-1], dtype=float).reshape(6,)

        # Pull last LKF posterior covariance
        # (your LKF returns P_meas and also aliases Phat_meas)
        P_hist = lkf_out.get("P_meas", None)
        if P_hist is None:
            P_hist = lkf_out.get("Phat_meas", None)
        if P_hist is None:
            raise KeyError("LKF output is missing P_meas/Phat_meas needed for warmstart.")

        P_start = np.asarray(P_hist[-1], dtype=float).reshape(6, 6)

        t_start = float(lkf_out["t_meas"][-1])

        # ------------------------------------------------------------
        # 2) Initialize EKF at LKF posterior
        # ------------------------------------------------------------
        self.Xhat = X_start.copy()
        self.Phat = P_start.copy()

        # If there’s nothing left to run, just return the LKF output
        if len(rest_meas) == 0:
            # make sure keys match the pipeline expectations
            out_comb = dict(lkf_out)
            out_comb["xhat_meas"] = out_comb.get("xhat_meas", out_comb["Xhat_meas"])
            out_comb["rms_final"] = out_comb.get("rms_final", None)
            out_comb["rms_by_iter"] = None
            out_comb["lkf_init"] = lkf_out
            out_comb["ekf"] = None
            out_comb["t_start_ekf"] = t_start
            return out_comb

        # ------------------------------------------------------------
        # 3) Run EKF on remaining chunk starting from t_start
        # ------------------------------------------------------------
        ekf_out = self.run(rest_meas, stations, Xtrue_meas=Xtrue_rest, t_prev_init=t_start)

        # ------------------------------------------------------------
        # 4) Stitch outputs into ONE dict (batch-plot compatible)
        # ------------------------------------------------------------
        t_comb = np.hstack([lkf_out["t_meas"], ekf_out["t_meas"]])
        station_comb = list(lkf_out["station_meas"]) + list(ekf_out["station_meas"])

        Xhat_comb = np.vstack([lkf_out["Xhat_meas"], ekf_out["Xhat_meas"]])

        P_meas_lkf = lkf_out.get("P_meas", lkf_out.get("Phat_meas"))
        P_meas_ekf = ekf_out.get("P_meas", None)
        if P_meas_ekf is None:
            raise KeyError("EKF output is missing P_meas; needed for combined plots.")
        P_meas_comb = np.concatenate([P_meas_lkf, P_meas_ekf], axis=0)

        two_sigma_comb = np.vstack([lkf_out["two_sigma_meas"], ekf_out["two_sigma_meas"]])

        prefit_comb = np.vstack([lkf_out["prefit_resids_final"], ekf_out["prefit_resids_final"]])
        postfit_lin_comb = np.vstack([lkf_out["postfit_resids_linear_final"], ekf_out["postfit_resids_linear_final"]])
        postfit_nl_comb = np.vstack([lkf_out["postfit_resids_meas"], ekf_out["postfit_resids_meas"]])

        Ppf_lkf = lkf_out.get("P_pf", None)
        Ppf_ekf = ekf_out.get("P_pf", None)
        if Ppf_lkf is None or Ppf_ekf is None:
            # not strictly required for your plots, but nice to keep consistent
            P_pf_comb = None
        else:
            P_pf_comb = np.vstack([Ppf_lkf, Ppf_ekf])

        if Xtrue_meas is None:
            err_comb = None
        else:
            err_comb = np.vstack([lkf_out["state_error_meas"], ekf_out["state_error_meas"]])

        # Combined RMS using your helper (keeps same structure as batch)
        rms_combined = KalmanFilterBase.compute_rms_summary_from_arrays(
            t=t_comb,
            postfit=postfit_nl_comb,
            state_err=err_comb,
            first_pass_gap_s=self.first_pass_gap_s,
        )

        return {
            "lkf_init": lkf_out,
            "ekf": ekf_out,
            "t_start_ekf": t_start,

            # batch-plot compatible keys
            "t_meas": t_comb,
            "station_meas": station_comb,
            "Xhat_meas": Xhat_comb,
            "xhat_meas": Xhat_comb,   # alias for your post_processing dataclass
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
