from __future__ import annotations

import time
import numpy as np

from .propagation import PropSettings, propagate_x_phi_step
from .range_rangerate import H_range_rangerate, H_tilde_range_rangerate_augmented
from .snc import state_noise_compensation, eci_to_ric_rotation
from .srp_dyn_model_man_estimate import apply_maneuver_jump


def _get_meas_pair_state_units(m: dict) -> np.ndarray:
    if ("rho_km" in m) and ("rho_dot_km_s" in m):
        return np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)
    if ("rho_m" in m) and ("rho_dot_m_s" in m):
        return np.array([m["rho_m"], m["rho_dot_m_s"]], dtype=float)
    raise KeyError(
        "Measurement dict must contain either "
        "('rho_km','rho_dot_km_s') or ('rho_m','rho_dot_m_s')."
    )


class IEKF2:
    """
    Size-agnostic EKF/IEKF with optional RTS smoother.

    State and measurements must be in consistent units.
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        R: np.ndarray,
        Q: np.ndarray,
        dyn_fun,
        dyn_jac,
        *,
        prop_settings: PropSettings | None = None,
        station_state_map: dict | None = None,
        station_start_index: int = 9,
        num_stations: int = 3,
        omega_vec: np.ndarray | None = None,
        q_frame: str = "eci",
        first_pass_gap_s: float = 6 * 3600.0,
    ):
        self.Xhat = np.asarray(x0, dtype=float).reshape(-1).copy()
        self.Phat = np.asarray(P0, dtype=float).copy()
        self.P0 = np.asarray(P0, dtype=float).copy()
        self.R = np.asarray(R, dtype=float).copy()
        self.Q = np.asarray(Q, dtype=float).copy()
        self.n = int(self.Xhat.size)

        if self.Phat.shape != (self.n, self.n):
            raise ValueError(f"P0 must be ({self.n},{self.n}); got {self.Phat.shape}")
        if self.R.shape != (2, 2):
            raise ValueError(f"R must be (2,2); got {self.R.shape}")
        if self.Q.shape not in {(3, 3), (6, 6), (self.n, self.n)}:
            raise ValueError(
                f"Q must be (3,3), (6,6), or ({self.n},{self.n}); got {self.Q.shape}"
            )

        self.dyn_fun = dyn_fun
        self.dyn_jac = dyn_jac
        self.prop_settings = PropSettings() if prop_settings is None else prop_settings

        self.station_state_map = station_state_map
        self.station_start_index = int(station_start_index)
        self.num_stations = int(num_stations)
        self.omega_vec = (
            np.array([0.0, 0.0, 7.2921158553e-5], dtype=float)
            if omega_vec is None
            else np.asarray(omega_vec, dtype=float).reshape(3)
        )
        self.q_frame = str(q_frame).strip().lower()
        self.first_pass_gap_s = float(first_pass_gap_s)

        self.hist = {"t": [], "postfit": [], "state_err": []}

    def reset_history(self):
        self.hist = {"t": [], "postfit": [], "state_err": []}

    def log_epoch(self, t, postfit_resid, Xtrue=None):
        self.hist["t"].append(float(t))
        self.hist["postfit"].append(np.asarray(postfit_resid, dtype=float).reshape(-1))
        if Xtrue is not None:
            err = self.Xhat - np.asarray(Xtrue, dtype=float).reshape(-1)
            self.hist["state_err"].append(err)

    def build_process_noise(
        self, delta_t: float, r_eci: np.ndarray | None = None, v_eci: np.ndarray | None = None
    ) -> np.ndarray:
        n = self.n
        if self.Q.shape == (n, n):
            return self.Q
        if n < 6:
            raise ValueError("State size < 6 requires full-shape Q=(n,n).")

        dt = max(float(delta_t), 0.0)
        if dt == 0.0:
            return np.zeros((n, n), dtype=float)

        if self.Q.shape == (3, 3):
            if self.q_frame == "eci":
                Q_accel_eci = self.Q
            elif self.q_frame == "ric":
                if r_eci is None or v_eci is None:
                    raise ValueError("RIC q_frame requires r_eci and v_eci.")
                C_eci_to_ric = eci_to_ric_rotation(r_eci=r_eci, v_eci=v_eci)
                C_ric_to_eci = C_eci_to_ric.T
                Q_accel_eci = C_ric_to_eci @ self.Q @ C_ric_to_eci.T
            else:
                raise ValueError(f"Unsupported q_frame '{self.q_frame}'.")
            Q6 = state_noise_compensation(delta_t=dt, n=6, m=3, Q=Q_accel_eci)
        elif self.Q.shape == (6, 6):
            Q6 = self.Q
        else:
            raise ValueError(f"Unsupported Q shape {self.Q.shape}")

        if n == 6:
            return Q6
        # Pad zeros for any extra states (Δv, Cr, etc.) — they are constants,
        # no process noise on them
        Qk = np.zeros((n, n), dtype=float)
        Qk[:6, :6] = Q6
        return Qk

    def G(self, station_or_key, X: np.ndarray, t: float):
        X = np.asarray(X, dtype=float).reshape(-1)
        r_sc = X[0:3]
        v_sc = X[3:6]

        if self.station_state_map is not None:
            st_idx = self.station_state_map[station_or_key]
            st_start = self.station_start_index + 3 * int(st_idx)
            r_gs = X[st_start:st_start + 3]
            v_gs = np.cross(self.omega_vec, r_gs)
            dr = r_sc - r_gs
            rho = np.linalg.norm(dr)
            if rho <= 0.0 or not np.isfinite(rho):
                return None
            rho_dot = float(np.dot(dr, (v_sc - v_gs)) / rho)
            if not np.isfinite(rho_dot):
                return None
            return np.array([rho, rho_dot], dtype=float)

        st_obj = station_or_key
        d = st_obj.measure(r_sc, v_sc, float(t))
        if d is None:
            return None
        rho = d["rho_km"] if "rho_km" in d else d["rho"]
        rho_dot = d["rho_dot_km_s"] if "rho_dot_km_s" in d else d["rho_dot"]
        if (not np.isfinite(rho)) or (not np.isfinite(rho_dot)):
            return None
        return np.array([rho, rho_dot], dtype=float)

    def _build_H(self, station_or_key, X_lin: np.ndarray, t: float) -> np.ndarray:
        if self.station_state_map is not None:
            st_idx = self.station_state_map[station_or_key]
            return H_tilde_range_rangerate_augmented(
                state=X_lin,
                stat_idx=int(st_idx),
                omega_rad_s=float(self.omega_vec[2]),
                station_start_index=self.station_start_index,
                num_stations=self.num_stations,
            )

        st_obj = station_or_key
        r_gs, v_gs = st_obj.station_eci(float(t))
        H_sc = H_range_rangerate(X_lin[:3], X_lin[3:6], r_gs, v_gs)

        # H_sc is (2,6) — pad zeros for any extra states (Δv cols 6:8, etc.)
        H = np.zeros((2, self.n), dtype=float)
        H[:, :6] = H_sc
        # columns 6:n remain zero — Δv has no direct measurement dependence
        return H

    @staticmethod
    def compute_rms_summary_from_arrays(t, postfit, state_err=None, first_pass_gap_s=6 * 3600.0):
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
        }

    def run(
        self,
        all_meas,
        stations=None,
        Xtrue_meas=None,
        t_prev_init=None,
        *,
        iterated: bool = False,
        max_iter: int = 10,
        iter_tol: float = 1.0e-8,
        bound_level: float = 5.0,
        no_snc_on_first_after_gap: bool = False,
        gap_threshold_s: float | None = None,
        show_progress: bool = False,
        progress_every: int = 50,
        t_man: float | None = None,   # maneuver epoch in seconds — set to None for no maneuver
    ) -> dict:
        """
        Run the IEKF/EKF forward pass.

        Parameters
        ----------
        t_man : float or None
            Maneuver epoch in seconds (same time system as your measurements).
            When set, the propagation step that straddles this epoch is split:
              1. propagate prev_time -> t_man
              2. apply instantaneous Δv kick  (apply_maneuver_jump)
              3. propagate t_man -> t
              4. chain the two STMs
            Set to None (default) to run without maneuver estimation — this
            keeps all existing 6-state or 7-state runs working unchanged.

        Eyeballing t_man
        ----------------
        Run a plain 6-state filter first, plot the postfit residuals, and look
        for the epoch where they suddenly jump or change sign.  Use that time
        as t_man.  The filter will then estimate the Δv that explains the jump.
        """
        station_map = {}
        if stations is not None:
            station_map = {st.name: st for st in stations}

        all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
        N = len(all_meas)
        if N == 0:
            raise ValueError("all_meas is empty.")

        t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
        st_meas = [m["station"] for m in all_meas]

        if Xtrue_meas is not None:
            Xtrue_meas = np.asarray(Xtrue_meas, dtype=float)
            if Xtrue_meas.shape != (N, self.n):
                raise ValueError(f"Xtrue_meas must have shape ({N}, {self.n}).")

        X_pf = np.full((N, self.n), np.nan, dtype=float)
        P_meas = np.full((N, self.n, self.n), np.nan, dtype=float)
        P_pf = np.full((N, self.n * self.n), np.nan, dtype=float)
        two_sigma = np.full((N, self.n), np.nan, dtype=float)

        prefit = np.full((N, 2), np.nan, dtype=float)
        postfit_lin = np.full((N, 2), np.nan, dtype=float)
        postfit_nl = np.full((N, 2), np.nan, dtype=float)
        nis_hist = np.full(N, np.nan, dtype=float)
        iter_counts = np.zeros(N, dtype=int)

        X_pred_hist = np.full((N, self.n), np.nan, dtype=float)
        P_pred_hist = np.full((N, self.n, self.n), np.nan, dtype=float)
        Phi_step_hist = np.full((N, self.n, self.n), np.nan, dtype=float)
        H_pred_hist = np.full((N, 2, self.n), np.nan, dtype=float)
        y_hist = np.full((N, 2), np.nan, dtype=float)
        meas_valid = np.zeros(N, dtype=bool)
        gap_step_mask = np.zeros(N, dtype=bool)
        snc_suppressed_mask = np.zeros(N, dtype=bool)

        state_error = None if Xtrue_meas is None else np.full((N, self.n), np.nan, dtype=float)

        X_hat = self.Xhat.copy()
        P = self.Phat.copy()
        prev_time = float(t_meas[0]) if t_prev_init is None else float(t_prev_init)

        # Track whether the maneuver jump has been applied yet so we only do it once
        maneuver_applied = False

        I_n = np.eye(self.n)
        I_2 = np.eye(2)
        sigma_bounds = float(bound_level) * np.sqrt(np.maximum(np.diag(self.R), 0.0))
        gap_thresh = self.first_pass_gap_s if gap_threshold_s is None else float(gap_threshold_s)
        t_wall = time.perf_counter()
        prog_step = max(int(progress_every), 1)

        for j in range(N):
            t = float(t_meas[j])
            st_key = st_meas[j]
            Y = _get_meas_pair_state_units(all_meas[j])
            y_hist[j, :] = Y
            have_meas = np.isfinite(Y).all()

            dt = t - prev_time

            # ------------------------------------------------------------------
            # PROPAGATION — three cases:
            #   A) zero step
            #   B) maneuver epoch falls in this interval (split propagation)
            #   C) normal propagation
            # ------------------------------------------------------------------
            if np.isclose(dt, 0.0):
                # Case A — no time has passed
                Xbar = X_hat.copy()
                Phi  = I_n.copy()

            elif (
                t_man is not None
                and not maneuver_applied
                and prev_time < t_man <= t
                and (t_man - prev_time) > 1.0e-9
            ):
                # Case B — maneuver epoch is inside this step
                #
                # Step 1: propagate from prev_time to t_man
                Xbar_man, Phi_man = propagate_x_phi_step(
                    x0=X_hat,
                    t0=prev_time,
                    t1=t_man,
                    f=self.dyn_fun,
                    A=self.dyn_jac,
                    settings=self.prop_settings,
                )

                # Step 2: apply instantaneous Δv kick and fill Phi Δv-sensitivity columns
                #   v+  = v-  + Δv
                #   Phi[0:6, 6:9] = Phi_rv @ [0; I]
                Xbar_man, Phi_man = apply_maneuver_jump(Xbar_man, Phi_man)
                maneuver_applied = True

                # Step 3: propagate from t_man to t using kicked state
                Xbar_k, Phi_k = propagate_x_phi_step(
                    x0=Xbar_man,
                    t0=t_man,
                    t1=t,
                    f=self.dyn_fun,
                    A=self.dyn_jac,
                    settings=self.prop_settings,
                )

                # Step 4: chain STMs  Phi(t, t0) = Phi(t, t_man) @ Phi(t_man, t0)
                Xbar = Xbar_k
                Phi  = Phi_k @ Phi_man

            else:
                # Case C — normal step, no maneuver
                Xbar, Phi = propagate_x_phi_step(
                    x0=X_hat,
                    t0=prev_time,
                    t1=t,
                    f=self.dyn_fun,
                    A=self.dyn_jac,
                    settings=self.prop_settings,
                )
            # ------------------------------------------------------------------

            is_gap_step = (j > 0) and np.isfinite(gap_thresh) and (dt > gap_thresh)
            gap_step_mask[j] = is_gap_step
            if no_snc_on_first_after_gap and is_gap_step:
                Qk = np.zeros((self.n, self.n), dtype=float)
                snc_suppressed_mask[j] = True
            else:
                Qk = self.build_process_noise(dt, r_eci=Xbar[:3], v_eci=Xbar[3:6])
            Pbar = Phi @ P @ Phi.T + Qk
            Pbar = 0.5 * (Pbar + Pbar.T)

            X_pred_hist[j, :] = Xbar
            P_pred_hist[j, :, :] = Pbar
            Phi_step_hist[j, :, :] = Phi

            st_rep = st_key if self.station_state_map is not None else station_map[st_key]
            Cbar = self.G(st_rep, Xbar, t) if have_meas else None

            if (Cbar is not None) and np.isfinite(Cbar).all():
                meas_valid[j] = True
                OminusC = Y - Cbar
                prefit[j, :] = OminusC
                Hbar = self._build_H(st_rep, Xbar, t)
                H_pred_hist[j, :, :] = Hbar

                if not iterated:
                    S = Hbar @ Pbar @ Hbar.T + self.R
                    K = Pbar @ Hbar.T @ np.linalg.solve(S, I_2)
                    X_hat = Xbar + K @ OminusC
                    A = I_n - K @ Hbar
                    P = A @ Pbar @ A.T + K @ self.R @ K.T
                    P = 0.5 * (P + P.T)
                    iter_counts[j] = 1
                    S_last = S
                else:
                    X_i = Xbar.copy()
                    eta = np.zeros(self.n, dtype=float)
                    K_last = np.zeros((self.n, 2), dtype=float)
                    H_last = Hbar.copy()
                    S_last = Hbar @ Pbar @ Hbar.T + self.R
                    taken = 0

                    for it in range(int(max_iter)):
                        H_i = self._build_H(st_rep, X_i, t)
                        h_i = self.G(st_rep, X_i, t)
                        if h_i is None or (not np.isfinite(h_i).all()):
                            break

                        y_res = Y - h_i

                        S = H_i @ Pbar @ H_i.T + self.R
                        K = Pbar @ H_i.T @ np.linalg.solve(S, I_2)
                        eta_new = K @ (y_res + H_i @ eta)

                        K_last = K
                        H_last = H_i
                        S_last = S

                        X_next = Xbar + eta_new
                        taken += 1
                        residual_ok = np.all(np.abs(y_res) <= sigma_bounds)
                        state_ok = np.linalg.norm(eta_new - eta) < float(iter_tol)
                        if it > 0 and (residual_ok or state_ok):
                            eta = eta_new
                            X_i = X_next
                            break

                        eta = eta_new
                        X_i = X_next

                    X_hat = X_i
                    A = I_n - K_last @ H_last
                    P = A @ Pbar @ A.T + K_last @ self.R @ K_last.T
                    P = 0.5 * (P + P.T)
                    iter_counts[j] = max(taken, 1)

                dx = X_hat - Xbar
                postfit_lin[j, :] = OminusC - (Hbar @ dx)

                Cpost = self.G(st_rep, X_hat, t)
                if Cpost is not None and np.isfinite(Cpost).all():
                    postfit_nl[j, :] = Y - Cpost
                    nis_hist[j] = float(postfit_nl[j, :].T @ np.linalg.solve(S_last, postfit_nl[j, :]))
            else:
                X_hat = Xbar
                P = Pbar

            X_pf[j, :] = X_hat
            P_meas[j, :, :] = P
            P_pf[j, :] = P.reshape(-1, order="F")
            two_sigma[j, :] = 2.0 * np.sqrt(np.maximum(np.diag(P), 0.0))

            if state_error is not None:
                state_error[j, :] = X_hat - Xtrue_meas[j, :]

            self.Xhat = X_hat.copy()
            self.Phat = P.copy()
            self.log_epoch(
                t,
                postfit_resid=postfit_nl[j, :],
                Xtrue=(None if Xtrue_meas is None else Xtrue_meas[j, :]),
            )

            prev_time = t

            if show_progress:
                k = j + 1
                if (k == N) or (k % prog_step == 0):
                    elapsed = max(time.perf_counter() - t_wall, 1e-9)
                    rate = k / elapsed
                    eta_t = (N - k) / rate if rate > 0.0 else float("inf")
                    bar_len = 28
                    n_fill = int(round(bar_len * k / max(N, 1)))
                    bar = "#" * n_fill + "-" * (bar_len - n_fill)
                    tag = "IEKF2" if iterated else "EKF2"
                    print(
                        f"\r{tag} Progress [{bar}] {k}/{N} ({100.0*k/max(N,1):5.1f}%) "
                        f"Elapsed {elapsed/60.0:6.2f} min  ETA {eta_t/60.0:6.2f} min",
                        end=("\n" if k == N else ""),
                        flush=True,
                    )

        rms_final = self.compute_rms_summary_from_arrays(
            t=t_meas,
            postfit=postfit_lin,
            state_err=state_error,
            first_pass_gap_s=self.first_pass_gap_s,
        )

        return {
            "t_meas": t_meas,
            "station_meas": st_meas,
            "xhat_meas": X_pf,
            "Xhat_meas": X_pf,
            "X_pf": X_pf,
            "state_error_meas": state_error,
            "prefit_resids_final": prefit,
            "postfit_resids_linear_final": postfit_lin,
            "postfit_resids_meas": postfit_nl,
            "P_meas": P_meas,
            "P_pf": P_pf,
            "two_sigma_meas": two_sigma,
            "X_pred_hist": X_pred_hist,
            "P_pred_hist": P_pred_hist,
            "Phi_step_hist": Phi_step_hist,
            "H_pred_hist": H_pred_hist,
            "y_hist": y_hist,
            "meas_valid": meas_valid,
            "gap_step_mask": gap_step_mask,
            "snc_suppressed_mask": snc_suppressed_mask,
            "nis_hist": nis_hist,
            "iter_counts": iter_counts,
            "rms_final": rms_final,
            "rms_by_iter": None,
            "R": self.R.copy(),
        }

    def smooth(self, run_out: dict, stations=None) -> dict:
        """
        RTS smoother based on forward EKF linearization history.
        """
        X_F = np.asarray(run_out["Xhat_meas"], dtype=float)
        P_F = np.asarray(run_out["P_meas"], dtype=float)
        X_bar = np.asarray(run_out["X_pred_hist"], dtype=float)
        P_bar = np.asarray(run_out["P_pred_hist"], dtype=float)
        Phi = np.asarray(run_out["Phi_step_hist"], dtype=float)
        prefit = np.asarray(run_out["prefit_resids_final"], dtype=float)
        H_hist = np.asarray(run_out.get("H_pred_hist"), dtype=float)
        t_meas = np.asarray(run_out["t_meas"], dtype=float)
        st_meas = list(run_out["station_meas"])
        y_hist = np.asarray(run_out.get("y_hist", np.full((len(t_meas), 2), np.nan)), dtype=float)
        meas_valid = np.asarray(run_out.get("meas_valid", np.isfinite(prefit).all(axis=1)), dtype=bool)

        N, n = X_F.shape
        I_n = np.eye(n)
        X_s = X_F.copy()
        P_s = P_F.copy()

        for k in range(N - 2, -1, -1):
            P_k_k = P_F[k]
            Phi_kp1_k = Phi[k + 1]
            P_kp1_k = P_bar[k + 1]
            Ck = np.linalg.solve(P_kp1_k, Phi_kp1_k @ P_k_k).T
            X_s[k] = X_F[k] + Ck @ (X_s[k + 1] - X_bar[k + 1])
            P_s[k] = P_F[k] + Ck @ (P_s[k + 1] - P_bar[k + 1]) @ Ck.T
            P_s[k] = 0.5 * (P_s[k] + P_s[k].T)

        two_sigma_smooth = 2.0 * np.sqrt(np.maximum(np.diagonal(P_s, axis1=1, axis2=2), 0.0))
        postfit_lin_smooth = np.full((N, 2), np.nan, dtype=float)
        for k in range(N):
            if (not meas_valid[k]) or (not np.all(np.isfinite(H_hist[k]))):
                continue
            postfit_lin_smooth[k, :] = prefit[k, :] - H_hist[k] @ (X_s[k] - X_bar[k])

        postfit_nl_smooth = np.full((N, 2), np.nan, dtype=float)
        station_map = {}
        if stations is not None:
            station_map = {st.name: st for st in stations}
        for k in range(N):
            if not meas_valid[k]:
                continue
            if self.station_state_map is not None:
                st_rep = st_meas[k]
            else:
                if not station_map:
                    break
                st_rep = station_map[st_meas[k]]
            C = self.G(st_rep, X_s[k], float(t_meas[k]))
            if C is not None and np.all(np.isfinite(C)):
                postfit_nl_smooth[k, :] = y_hist[k, :] - C

        state_error_smooth = None
        if run_out.get("state_error_meas", None) is not None:
            err_f = np.asarray(run_out["state_error_meas"], dtype=float)
            Xtrue = X_F - err_f
            state_error_smooth = X_s - Xtrue

        rms_smooth = self.compute_rms_summary_from_arrays(
            t=t_meas,
            postfit=postfit_lin_smooth,
            state_err=state_error_smooth,
            first_pass_gap_s=self.first_pass_gap_s,
        )

        smooth_out = {
            "Xhat_smooth": X_s,
            "P_smooth": P_s,
            "two_sigma_smooth": two_sigma_smooth,
            "state_error_smooth_meas": state_error_smooth,
            "postfit_resids_linear_smooth": postfit_lin_smooth,
            "postfit_resids_smooth": postfit_lin_smooth,
            "postfit_resids_smooth_nl": postfit_nl_smooth,
            "rms_smooth": rms_smooth,
        }
        run_out.update(smooth_out)
        return smooth_out

    def smooth_fraser_potter(self, run_out: dict, stations=None) -> dict:
        """
        Fraser-Potter forward-backward EKF smoother.
 
        This is the smoother the lecture slides prescribe for the EKF
        (Lecture 19, slides 4-5).  It runs a backward EKF using the
        forward filter's stored linearization, then optimally combines
        the two estimates at every epoch.
 
        Algorithm
        ---------
        From the slides (slide 5 equivalent form):
 
            W_k  = P⁻_{B,k} (P⁺_{F,k} + P⁻_{B,k})⁻¹
            X̂_S  = W_k X̂_F  + (I - W_k) X̄_B
            P_S  = W_k P⁺_F  W_k^T + (I-W_k) P⁻_B (I-W_k)^T
 
        where:
            P⁺_{F,k}  = forward posterior  (uses data 0 … k)
            P⁻_{B,k}  = backward prior     (uses data k+1 … N)
            X̂_{F,k}   = forward posterior estimate
            X̄_{B,k}   = backward prior estimate
 
        Together they cover all data, giving the optimal smoothed estimate.
 
        Backward filter initial conditions (slide 4):
            X̄_{B,f} = X̂_{F,f}         (start from forward final estimate)
            P⁻_{B,f} = ∞   ⟹   Λ_{B,f} = 0  (no information from the future)
 
        Notes
        -----
        - The backward time update uses Φ⁻¹ to propagate covariance backward.
          This is numerically well-conditioned for short steps but may
          accumulate error over very long gaps.
        - Process noise is omitted in the backward time update — this is the
          standard approximation and is accurate when Q is small relative to P.
        - At the final epoch, the smoothed estimate equals the forward estimate
          (W_N → I because P⁻_{B,N} = ∞).
        """
        X_F    = np.asarray(run_out["Xhat_meas"],    dtype=float)   # forward posteriors
        P_F    = np.asarray(run_out["P_meas"],        dtype=float)   # forward posterior covs
        Phi    = np.asarray(run_out["Phi_step_hist"], dtype=float)   # Phi[k] = Phi(t_k, t_{k-1})
        H_hist = np.asarray(run_out["H_pred_hist"],   dtype=float)   # measurement Jacobians
        y_hist_arr = np.asarray(
            run_out.get("y_hist", np.full((len(run_out["t_meas"]), 2), np.nan)),
            dtype=float,
        )
        meas_valid = np.asarray(run_out["meas_valid"], dtype=bool)
        prefit     = np.asarray(run_out["prefit_resids_final"], dtype=float)
        t_meas     = np.asarray(run_out["t_meas"], dtype=float)
        st_meas    = list(run_out["station_meas"])
 
        N, n = X_F.shape
        I_n  = np.eye(n)
        I_2  = np.eye(2)
 
        # ----------------------------------------------------------------
        # Backward filter pass
        # Storage: X_B_prior[k], P_B_prior[k] are the backward PRIOR at k
        # (i.e. the estimate using data from k+1 onward, before seeing y_k)
        # ----------------------------------------------------------------
        X_B_prior = np.full((N, n),    np.nan, dtype=float)
        P_B_prior = np.full((N, n, n), np.nan, dtype=float)
 
        # Initial conditions — last epoch
        # P⁻_{B,f} = ∞ represented by a very large diagonal
        # X̄_{B,f}  = X̂_{F,f}  (mean doesn't matter when covariance is ∞)
        INF_COV = 1.0e16
        X_B_cur = X_F[N - 1].copy()
        P_B_cur = INF_COV * I_n
 
        X_B_prior[N - 1] = X_B_cur.copy()
        P_B_prior[N - 1] = P_B_cur.copy()
 
        # Sweep backward: k = N-1 down to 1
        for k in range(N - 1, 0, -1):
 
            # ---- Backward measurement update at k ----
            # We use the same H linearized at the forward predicted state
            # (the standard EKF approach for the backward pass)
            if meas_valid[k] and np.all(np.isfinite(H_hist[k])):
                H_k = H_hist[k]            # (2, n)
                y_k = y_hist_arr[k]        # (2,)
 
                S_B  = H_k @ P_B_cur @ H_k.T + self.R
                K_B  = P_B_cur @ H_k.T @ np.linalg.solve(S_B, I_2)
                X_B_post = X_B_cur + K_B @ (y_k - H_k @ X_B_cur)
                A_B      = I_n - K_B @ H_k
                P_B_post = A_B @ P_B_cur @ A_B.T + K_B @ self.R @ K_B.T
                P_B_post = 0.5 * (P_B_post + P_B_post.T)
            else:
                X_B_post = X_B_cur.copy()
                P_B_post = P_B_cur.copy()
 
            # ---- Backward time update: propagate posterior at k → prior at k-1 ----
            # x_{k-1} ≈ Phi(t_k, t_{k-1})^{-1} x_k
            # Phi[k] stores Phi(t_k, t_{k-1}), so we need its inverse
            Phi_k = Phi[k]   # (n, n)  Phi from t_{k-1} to t_k
 
            # Use solve for numerical stability: Phi_k @ x_{k-1} = x_k
            X_B_cur = np.linalg.solve(Phi_k, X_B_post)
 
            # P_{B,k-1}^- = Phi_k^{-1} P_{B,k}^+ Phi_k^{-T}
            # Equivalent to: solve(Phi_k, solve(Phi_k, P_B_post).T).T
            Phi_inv_P = np.linalg.solve(Phi_k, P_B_post)          # Phi^{-1} P
            P_B_cur   = np.linalg.solve(Phi_k, Phi_inv_P.T).T     # Phi^{-1} P Phi^{-T}
            P_B_cur   = 0.5 * (P_B_cur + P_B_cur.T)
 
            # Store as prior for epoch k-1
            X_B_prior[k - 1] = X_B_cur.copy()
            P_B_prior[k - 1] = P_B_cur.copy()
 
        # ----------------------------------------------------------------
        # Combination: Fraser-Potter equations (slide 5)
        #
        #   W_k  = P⁻_B (P⁺_F + P⁻_B)^{-1}
        #   X̂_S  = W X̂_F + (I-W) X̄_B
        #   P_S  = W P⁺_F W^T + (I-W) P⁻_B (I-W)^T
        # ----------------------------------------------------------------
        X_s = np.full((N, n),    np.nan, dtype=float)
        P_s = np.full((N, n, n), np.nan, dtype=float)
 
        for k in range(N):
            PF = P_F[k]           # forward posterior  P⁺_{F,k}
            PB = P_B_prior[k]     # backward prior     P⁻_{B,k}
            XF = X_F[k]
            XB = X_B_prior[k]
 
            if not np.all(np.isfinite(PB)) or not np.all(np.isfinite(XB)):
                # Fallback: forward only (shouldn't happen except at k=N-1)
                X_s[k] = XF.copy()
                P_s[k] = PF.copy()
                continue
 
            # W_k = P⁻_B (P⁺_F + P⁻_B)^{-1}
            # Solve: (P⁺_F + P⁻_B) W_k^T = P⁻_B^T  → W_k = solution^T
            W = np.linalg.solve((PF + PB).T, PB.T).T    # (n, n)
 
            ImW = I_n - W
            X_s[k] = W @ XF + ImW @ XB
            P_s[k] = W @ PF @ W.T + ImW @ PB @ ImW.T
            P_s[k] = 0.5 * (P_s[k] + P_s[k].T)
 
        # ----------------------------------------------------------------
        # Smoothed residuals and post-processing
        # ----------------------------------------------------------------
        two_sigma_fp = 2.0 * np.sqrt(np.maximum(np.diagonal(P_s, axis1=1, axis2=2), 0.0))
 
        postfit_lin_fp = np.full((N, 2), np.nan, dtype=float)
        for k in range(N):
            if (not meas_valid[k]) or (not np.all(np.isfinite(H_hist[k]))):
                continue
            # Linearized postfit using the smoothed state
            X_pred_k = np.asarray(run_out["X_pred_hist"][k], dtype=float)
            postfit_lin_fp[k, :] = prefit[k, :] - H_hist[k] @ (X_s[k] - X_pred_k)
 
        postfit_nl_fp = np.full((N, 2), np.nan, dtype=float)
        station_map = {}
        if stations is not None:
            station_map = {st.name: st for st in stations}
        for k in range(N):
            if not meas_valid[k]:
                continue
            st_rep = st_meas[k] if self.station_state_map is not None else station_map.get(st_meas[k])
            if st_rep is None:
                break
            C = self.G(st_rep, X_s[k], float(t_meas[k]))
            if C is not None and np.all(np.isfinite(C)):
                postfit_nl_fp[k, :] = y_hist_arr[k, :] - C
 
        state_error_fp = None
        if run_out.get("state_error_meas") is not None:
            err_f  = np.asarray(run_out["state_error_meas"], dtype=float)
            Xtrue  = X_F - err_f
            state_error_fp = X_s - Xtrue
 
        rms_fp = self.compute_rms_summary_from_arrays(
            t=t_meas, postfit=postfit_lin_fp,
            state_err=state_error_fp, first_pass_gap_s=self.first_pass_gap_s,
        )
 
        fp_out = {
            "Xhat_smooth_fp": X_s,
            "P_smooth_fp": P_s,
            "two_sigma_smooth_fp": two_sigma_fp,
            "X_B_prior": X_B_prior,
            "P_B_prior": P_B_prior,
            "state_error_smooth_fp": state_error_fp,
            "postfit_resids_linear_smooth_fp": postfit_lin_fp,
            "postfit_resids_smooth_fp": postfit_nl_fp,
            "rms_smooth_fp": rms_fp,
        }
        run_out.update(fp_out)
        return fp_out
