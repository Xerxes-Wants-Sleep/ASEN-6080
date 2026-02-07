import numpy as np
from .propagation import propagate_x_phi_history, PropSettings
from .range_rangerate import H_range_rangerate, H_tilde_range_rangerate_augmented

## Batch Propagator for ASEN 6080 Project 1

def batch_estimate_x0(
    all_meas,
    stations,
    x0_bar: np.ndarray,
    P0: np.ndarray,
    R: np.ndarray,
    dyn_fun,
    dyn_jac,
    station_state_map: dict | None = None,   # {'StationName': 0/1/2} or {101:0,...} depending on your meas field
    max_iter: int = 10,
    tol: float = 1e-8,
    prop_settings: PropSettings | None = None,
    station_start_index: int = 9,
    num_stations: int = 3,
    omega_vec: np.ndarray | None = None,
):
    """
    Iterated nonlinear batch LS.
    - 6-state mode: stations come from station objects.
    - 18-state mode: stations come from the state vector (and rotate in dynamics).
    """

    if prop_settings is None:
        prop_settings = PropSettings()

    # Earth rotation vector (must match what your dynamics uses)
    if omega_vec is None:
        omega_vec = np.array([0.0, 0.0, 7.2921158553e-5], dtype=float)
    else:
        omega_vec = np.asarray(omega_vec, dtype=float).reshape(3)

    # Map station objects by name (only used in 6-state mode)
    station_obj_map = {st.name: st for st in stations}

    # Sort and unpack times
    all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
    st_meas = [m["station"] for m in all_meas]
    mcount = len(all_meas)
    t0 = float(t_meas[0])

    x0_bar = np.asarray(x0_bar, dtype=float).reshape(-1)
    n = x0_bar.size
    x0_star = x0_bar.copy()

    P0inv = np.linalg.inv(P0)
    Rinv = np.linalg.inv(R)

    dx0_hist = []
    x0_star_hist = [x0_star.copy()]
    prefit_resids_hist = []
    postfit_resids_linear_hist = []
    Lambda_hist = []

    # ---- baseline trajectory for *prefit* residuals (friend-style) ----
    X_base, Phi_base = propagate_x_phi_history(
        x0=x0_bar,
        t_eval=t_meas,
        f=dyn_fun,
        A=dyn_jac,
        settings=prop_settings
    )

    # -----------------------------
    # 1) Iterated batch LS
    # -----------------------------
    Lambda = None
    N_vec = None

    for it in range(max_iter):
        Lambda = P0inv.copy()

        # matches friend: dx_bar = X_0_guess - X_ref_0
        dx_bar = x0_bar - x0_star
        N_vec = P0inv @ dx_bar

        # Propagate current reference state + STM
        X_hist, Phi_hist = propagate_x_phi_history(
            x0=x0_star,
            t_eval=t_meas,
            f=dyn_fun,
            A=dyn_jac,
            settings=prop_settings
        )

        # residual storage for this iteration (mostly for debugging)
        OminusC = np.full((mcount, 2), np.nan, dtype=float)
        H_store = np.full((mcount, 2, n), np.nan, dtype=float)

        rms_acum_range = 0.0
        rms_acum_rr = 0.0

        for j, m in enumerate(all_meas):
            ti = float(m["t"])
            st_key = m["station"]  # could be name or ID depending on your dataset

            x_curr = X_hist[j, :]
            Phi_curr = Phi_hist[j, :, :]

            # --- observed measurement (standardize to meters + m/s) ---
            # If your meas dict uses km fields, convert here.
            if "rho_km" in m:
                y_obs = np.array([m["rho_km"] * 1000.0, m["rho_dot_km_s"] * 1000.0], dtype=float)
            else:
                # assume meters
                y_obs = np.array([m["rho_m"], m["rho_dot_m_s"]], dtype=float)

            # --- predicted measurement + H_local ---
            if (n >= 18) and (station_state_map is not None):
                # === 18-state mode (friend-style) ===
                st_idx = station_state_map[st_key]  # 0/1/2

                st_start = station_start_index + 3 * st_idx
                r_gs = x_curr[st_start:st_start+3]
                v_gs = np.cross(omega_vec, r_gs)

                r_sc = x_curr[0:3]
                v_sc = x_curr[3:6]

                dr = r_sc - r_gs
                rho = np.linalg.norm(dr)
                if rho <= 0.0:
                    continue
                rho_dot = float(np.dot(dr, (v_sc - v_gs)) / rho)

                y_comp = np.array([rho, rho_dot], dtype=float)

                # Local H wrt current state (2×n)
                H_local = H_tilde_range_rangerate_augmented(
                    state=x_curr,
                    stat_idx=st_idx,
                    station_start_index=station_start_index,
                    num_stations=num_stations
                )

            else:
                # === 6-state (or non-station augmented) mode ===
                st_obj = station_obj_map[st_key]
                r_sc = x_curr[0:3]
                v_sc = x_curr[3:6]

                # Station from object
                r_st, v_st = st_obj.station_eci(ti)

                # predict
                dr = r_sc - r_st
                rho = np.linalg.norm(dr)
                if rho <= 0.0:
                    continue
                rho_dot = float(np.dot(dr, (v_sc - v_st)) / rho)
                y_comp = np.array([rho, rho_dot], dtype=float)

                H_sc = H_range_rangerate(r_sc, v_sc, r_st, v_st)
                H_local = np.zeros((2, n), dtype=float)
                H_local[:, :6] = H_sc

            resid = y_obs - y_comp
            OminusC[j, :] = resid

            # Map local H to epoch: H_epoch = H_local @ Phi(tk,t0)
            H_epoch = H_local @ Phi_curr
            H_store[j, :, :] = H_epoch

            # Accumulate normal equations
            Lambda += H_epoch.T @ Rinv @ H_epoch
            N_vec += H_epoch.T @ Rinv @ resid

            rms_acum_range += resid[0] ** 2
            rms_acum_rr += resid[1] ** 2

        # Save normal matrix for this iteration (for covariance history)
        Lambda_hist.append(Lambda.copy())

        # Solve for correction and update epoch state
        dx0 = np.linalg.solve(Lambda, N_vec)

        # Linearized postfit residuals: r_pf = r_pre - H*dx0
        resid_pf = np.full((mcount, 2), np.nan, dtype=float)
        valid = np.isfinite(OminusC).all(axis=1)
        if np.any(valid):
            resid_pf[valid, :] = OminusC[valid, :] - np.einsum("ijk,k->ij", H_store[valid, :, :], dx0)

        prefit_resids_hist.append(OminusC.copy())
        postfit_resids_linear_hist.append(resid_pf.copy())

        x0_star = x0_star + dx0

        dx0_hist.append(dx0.copy())
        x0_star_hist.append(x0_star.copy())

        dx_norm = np.linalg.norm(dx0)
        rms_range = np.sqrt(rms_acum_range / max(1, mcount))
        rms_rr = np.sqrt(rms_acum_rr / max(1, mcount))

        print(f"Batch Iter {it+1}: |dx| = {dx_norm:.6e} | RMS rho = {rms_range:.4f} m | RMS rhodot = {rms_rr:.5f} m/s")

        if dx_norm < tol:
            print("Batch converged.")
            break

    # -----------------------------
    # 2) Outputs at epoch
    # -----------------------------
    x0_hat = x0_star
    P0_hat = np.linalg.inv(Lambda)

    # -----------------------------
    # 3) Friend-style FINAL PASS for plotting data
    # -----------------------------
    X_hat, Phi_hat = propagate_x_phi_history(
        x0=x0_hat,
        t_eval=t_meas,
        f=dyn_fun,
        A=dyn_jac,
        settings=prop_settings
    )

    # Covariance history: P(tk) = Phi(tk,t0) P0 Phi^T
    P_hist = np.zeros((mcount, n, n), dtype=float)
    for k in range(mcount):
        Phi_k = Phi_hat[k, :, :]
        P_hist[k, :, :] = Phi_k @ P0_hat @ Phi_k.T

    prefit = np.full((mcount, 2), np.nan, dtype=float)
    postfit = np.full((mcount, 2), np.nan, dtype=float)
    nis_hist = np.full(mcount, np.nan, dtype=float)

    for k, m in enumerate(all_meas):
        st_key = m["station"]

        if "rho_km" in m:
            y_obs = np.array([m["rho_km"] * 1000.0, m["rho_dot_km_s"] * 1000.0], dtype=float)
        else:
            y_obs = np.array([m["rho_m"], m["rho_dot_m_s"]], dtype=float)

        # --- baseline prediction ---
        xb = X_base[k, :]
        # --- final prediction ---
        xh = X_hat[k, :]

        if (n >= 18) and (station_state_map is not None):
            st_idx = station_state_map[st_key]
            st_start = station_start_index + 3 * st_idx

            def predict_from_state(xrow):
                r_gs = xrow[st_start:st_start+3]
                v_gs = np.cross(omega_vec, r_gs)
                r_sc = xrow[0:3]
                v_sc = xrow[3:6]
                dr = r_sc - r_gs
                rho = np.linalg.norm(dr)
                rho_dot = float(np.dot(dr, (v_sc - v_gs)) / rho)
                return np.array([rho, rho_dot], dtype=float)

            y_pre = predict_from_state(xb)
            y_post = predict_from_state(xh)

            res_pre = y_obs - y_pre
            res_post = y_obs - y_post

            H_local = H_tilde_range_rangerate_augmented(
                state=xh,
                stat_idx=st_idx,
                station_start_index=station_start_index,
                num_stations=num_stations
            )

        else:
            st_obj = station_obj_map[st_key]

            def predict6(xrow):
                r_sc = xrow[0:3]; v_sc = xrow[3:6]
                r_st, v_st = st_obj.station_eci(float(m["t"]))
                dr = r_sc - r_st
                rho = np.linalg.norm(dr)
                rho_dot = float(np.dot(dr, (v_sc - v_st)) / rho)
                return np.array([rho, rho_dot], dtype=float)

            y_pre = predict6(xb)
            y_post = predict6(xh)

            res_pre = y_obs - y_pre
            res_post = y_obs - y_post

            r_st, v_st = st_obj.station_eci(float(m["t"]))
            H_sc = H_range_rangerate(xh[0:3], xh[3:6], r_st, v_st)
            H_local = np.zeros((2, n), dtype=float)
            H_local[:, :6] = H_sc

        prefit[k, :] = res_pre
        postfit[k, :] = res_post

        S = H_local @ P_hist[k, :, :] @ H_local.T + R
        nis_hist[k] = float(res_post.T @ np.linalg.solve(S, res_post))

    # -----------------------------
    # 4) Info dict (superset of friend’s stored data)
    # -----------------------------
    info = {
        "t0": t0,
        "t_meas": t_meas,
        "station_meas": st_meas,

        # iteration-level
        "dx0_hist": dx0_hist,
        "x0_star_hist": x0_star_hist,
        "num_iters": len(dx0_hist),
        "prefit_resids_hist": prefit_resids_hist,                     # list of (m,2)
        "postfit_resids_linear_hist": postfit_resids_linear_hist,      # list of (m,2)
        "Lambda_hist": Lambda_hist,                                    # list of (n,n)

        # final normal eqns
        "Lambda_final": Lambda,
        "N_final": N_vec,

        # friend-style plot-ready histories
        "state_hist": X_hat,      # (mcount, n)
        "phi_hist": Phi_hat,      # (mcount, n, n)
        "P_hist": P_hist,         # (mcount, n, n)
        "prefit_residuals": prefit,   # (mcount, 2)
        "postfit_residuals": postfit, # (mcount, 2)
        "nis_hist": nis_hist,         # (mcount,)

        # extra: deviation vs baseline over time (friend’s dx_hist)
        "dx_hist_time": (X_hat - X_base),  # (mcount, n)
    }

    return x0_hat, P0_hat, info
