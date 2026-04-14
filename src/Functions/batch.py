import numpy as np
from scipy.integrate import solve_ivp
from .jacobians import stm, stm_generic
from .range_rangerate import H_range_rangerate, H_tilde_range_rangerate_augmented
from. propagation import propagate_x_phi_step, PropSettings


def propagate_state_and_stm_history(
    x0_ref_6: np.ndarray,
    t_eval: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    Re: float = 6378.0,
    reltol: float = 1e-10,
    abstol: float = 1e-10,
    method: str = "DOP853",
):
    """
    One integration from t_eval[0] to t_eval[-1], returning state+STM at all t_eval
    """
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    t0 = float(t_eval[0])
    tf = float(t_eval[-1])

    remove = np.array([6, 7, 8], dtype=int)
    nx = 6

    x0_ref_6 = np.asarray(x0_ref_6, dtype=float).reshape(6,)
    X0_9 = np.hstack((x0_ref_6, mu, J2, J3))
    Phi0 = np.eye(nx)
    y0 = np.hstack((X0_9, Phi0.flatten()))

    fun = lambda t, y: stm(
        t,
        state9=y[:9],
        phi=y[9:].reshape(nx, nx),
        rows_col_to_remove=remove,
        Re=Re,
        j2=True,
        j3=False,
    )

    sol = solve_ivp(
        fun, (t0, tf), y0,
        t_eval=t_eval,
        rtol=reltol, atol=abstol,
        method=method,
    )

    Y = sol.y.T                       # (N, 9+36)
    X_hist = Y[:, :6]                 # (N,6)
    Phi_hist = Y[:, 9:].reshape(-1, nx, nx)  # (N,6,6)

    return X_hist, Phi_hist



def predict_meas_vec(station, x_i_6, ti):
    d = station.measure(x_i_6[:3], x_i_6[3:6], float(ti))
    return np.array([d["rho_km"], d["rho_dot_km_s"]], dtype=float)



def batch_estimate_x0(
    all_meas,
    stations,
    x0_bar: np.ndarray,
    P0: np.ndarray,
    R: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    Re: float = 6378.0,
    max_iter: int = 10,
    tol: float = 1e-10,
    reltol: float = 1e-10,
    abstol: float = 1e-10,
    method: str = "DOP853",
    dyn_fun=None,
    dyn_jac=None,
    prop_settings: PropSettings | None = None,
):
    # -----------------------------
    # 0) Setup / bookkeeping
    # -----------------------------
    station_map = {st.name: st for st in stations}

    all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
    st_meas = [m["station"] for m in all_meas]
    mcount = len(all_meas)
    t0 = float(t_meas[0])

    x0_bar = np.asarray(x0_bar, dtype=float).reshape(-1)
    n = x0_bar.size

    # Generic n-state fallback (e.g., 7-state [r,v,Cr]) uses the newer batch path.
    if n != 6:
        if dyn_fun is None or dyn_jac is None:
            raise ValueError(
                "For non-6-state batch runs, provide dyn_fun and dyn_jac. "
                "These should match the state dimension of x0_bar."
            )
        from .batch_18_state import batch_estimate_x0 as batch_estimate_x0_generic

        # Keep legacy km-unit behavior when routing through generic batch path.
        all_meas_generic = []
        for m in all_meas:
            d = dict(m)
            if ("rho_km" in d) and ("rho_dot_km_s" in d):
                d["rho_m"] = d.pop("rho_km")
                d["rho_dot_m_s"] = d.pop("rho_dot_km_s")
            all_meas_generic.append(d)

        settings = prop_settings
        if settings is None:
            settings = PropSettings(rtol=reltol, atol=abstol, method=method)

        return batch_estimate_x0_generic(
            all_meas=all_meas_generic,
            stations=stations,
            x0_bar=x0_bar,
            P0=P0,
            R=R,
            dyn_fun=dyn_fun,
            dyn_jac=dyn_jac,
            max_iter=max_iter,
            tol=tol,
            prop_settings=settings,
        )

    x0_bar = x0_bar.reshape(6,)
    x0_star = x0_bar.copy()

    # Use these as-is for now (later we can swap to Cholesky solves)
    P0inv = np.linalg.inv(P0)
    Rinv = np.linalg.inv(R)

    dx0_hist = []
    x0_star_hist = [x0_star.copy()]

    prefit_resids_final = None
    postfit_resids_linear_final = None

    v_zero = np.zeros(3)

    # -----------------------------
    # 1) Iterated batch LS
    # -----------------------------
    for it in range(max_iter):

        # 1a) Initialize normal equations with a priori
        L = P0inv.copy()
        N = P0inv @ (x0_bar - x0_star)

        # 1b) Propagate reference + STM to measurement times
        X_hist, Phi_hist = propagate_state_and_stm_history(
            x0_ref_6=x0_star,
            t_eval=t_meas,
            mu=mu, J2=J2, J3=J3,
            Re=Re,
            reltol=reltol,
            abstol=abstol,
            method=method,
        )

        # 1c) Allocate aligned residual/Jacobian storage
        OminusC = np.full((mcount, 2), np.nan, dtype=float)     # prefit residuals
        H_store = np.full((mcount, 2, 6), np.nan, dtype=float)  # store H = Htilde*Phi

        # 1d) Measurement loop: build L, N
        for j, m in enumerate(all_meas):
            ti = float(m["t"])
            st = station_map[m["station"]]

            x_i = X_hist[j, :]
            Phi_i0 = Phi_hist[j, :, :]

            d = st.measure(x_i[:3], x_i[3:6], ti)
            if d is None:
                continue

            yhat = np.array([d["rho_km"], d["rho_dot_km_s"]], dtype=float)
            yobs = np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)

            resid = yobs - yhat
            OminusC[j, :] = resid

            Rs, Vs, _ = st.ecef2eci(ti, st.r_ecef, v_zero)

            Htilde = H_range_rangerate(x_i[:3], x_i[3:6], Rs, Vs)  # (2,6)
            H = Htilde @ Phi_i0                                     # (2,6)
            H_store[j, :, :] = H

            L += H.T @ Rinv @ H
            N += H.T @ Rinv @ resid

        # 1e) Solve for correction and update reference
        dx0 = np.linalg.solve(L, N)
        x0_star = x0_star + dx0

        dx0_hist.append(dx0.copy())
        x0_star_hist.append(x0_star.copy())

        # 1f) Linearized postfits (matches professor): r_pf = r_pre - H*dx0
        valid = np.isfinite(OminusC).all(axis=1)
        resid_pf = np.full((mcount, 2), np.nan, dtype=float)
        if np.any(valid):
            resid_pf[valid, :] = OminusC[valid, :] - np.einsum("ijk,k->ij", H_store[valid, :, :], dx0)

        # Save final-iteration residual arrays for plotting
        prefit_resids_final = OminusC
        postfit_resids_linear_final = resid_pf

        # 1g) Convergence check
        if np.linalg.norm(dx0) < tol:
            break

    # -----------------------------
    # 2) Outputs
    # -----------------------------
    x0_hat = x0_star
    P0_hat = np.linalg.inv(L)  # later we can do this via Cholesky/solve

    info = {
        "t0": t0,
        "t_meas": t_meas,
        "station_meas": st_meas,

        "dx0_hist": dx0_hist,
        "x0_star_hist": x0_star_hist,
        "num_iters": len(dx0_hist),

        "prefit_resids_final": prefit_resids_final,                  # (m,2), NaNs for masked
        "postfit_resids_linear_final": postfit_resids_linear_final,  # (m,2), NaNs for masked

        "Lambda_final": L,
        "N_final": N,
    }

    return x0_hat, P0_hat, info


