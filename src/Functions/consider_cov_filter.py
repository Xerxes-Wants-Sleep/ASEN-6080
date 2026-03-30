import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp

from .jacobians import consider_cov_partial_j3, stm
from .range_rangerate import H_range_rangerate


def sequential_consider_cov_analysis(
    X0_star: np.ndarray,
    all_meas: list[dict],
    stations: list,
    R: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    Re: float,
    P0: np.ndarray,
    P_cc0: np.ndarray,
    x0: np.ndarray | None = None,
    S0: np.ndarray | None = None,
    num_meas: int | None = None,
    reltol: float = 1e-10,
    abstol: float = 1e-10,
    method: str = "DOP853",
) -> dict:
    n = 6
    q = 1
    I_n = np.eye(n)
    remove = np.array([6, 7, 8], dtype=int)

    Xstar_prev = np.asarray(X0_star, dtype=float).reshape(n,)
    P_xk_prev = np.asarray(P0, dtype=float).reshape(n, n)
    P_cc0 = np.asarray(P_cc0, dtype=float).reshape(q, q)
    xhat_prev = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(n,)
    S_prev = np.zeros((n, q)) if S0 is None else np.asarray(S0, dtype=float).reshape(n, q)
    R = np.asarray(R, dtype=float)

    station_map = {st.name: st for st in stations}
    all_meas = sorted(all_meas, key=lambda m: float(m["t"]))

    t_epochs = []
    meas_epochs = []
    for m in all_meas:
        t = float(m["t"])
        if (len(t_epochs) == 0) or (t != t_epochs[-1]):
            t_epochs.append(t)
            meas_epochs.append([m])
        else:
            meas_epochs[-1].append(m)

    xEst = []
    xcEst = []
    PEst = []
    PcEst = []
    PxcEst = []
    prefit_res = []
    postfit_res = []
    postfit_res_c = []
    XEst = []
    XcEst = []
    Phi_total = []
    Theta_total = []
    Psi = []
    statVis = []
    iter_hist = []

    t_prev = float(t_epochs[0])
    Phi_tk_t0 = np.eye(n)
    theta_tk_t0 = np.zeros((n, q))
    
    if num_meas is None:
        num_meas = len(t_epochs) 

    for k in range(1, num_meas):
        t_i = float(t_epochs[k])
        meas_k = meas_epochs[k]

        y0_phi = np.hstack([Xstar_prev, np.array([mu, J2, J3]), I_n.reshape(-1)])
        sol_phi = solve_ivp(
            fun=lambda t, y: stm(
                t=t,
                state9=y[:9],
                phi=y[9:].reshape(n, n),
                rows_col_to_remove=remove,
                Re=Re,
                j2=True,
                j3=False,
            ),
            t_span=(t_prev, t_i),
            y0=y0_phi,
            t_eval=[t_i],
            rtol=reltol,
            atol=abstol,
            method=method,
        )
        y_phi_i = sol_phi.y[:, -1]
        Xstar_k = y_phi_i[:n]
        Phi_k_km1 = y_phi_i[9:].reshape(n, n)

        y0_theta = np.hstack([Xstar_prev, np.array([mu, J2, J3]), np.zeros(n * q)])

        def theta_eom(t: float, y: np.ndarray) -> np.ndarray:
            state9 = y[:9]
            S_tmp = y[9:].reshape(n, q)
            stm_out = stm(
                t=t,
                state9=state9,
                phi=I_n,
                rows_col_to_remove=remove,
                Re=Re,
                j2=True,
                j3=False,
            )
            dxdt = stm_out[:9]
            A = stm_out[9:].reshape(n, n)
            d = np.zeros((n, q))
            d[3:6, 0] = consider_cov_partial_j3(state9[:3], state9[6], state9[7], state9[8], Re)
            Sdot = A @ S_tmp + d
            return np.hstack([dxdt, Sdot.reshape(-1)])

        sol_theta = solve_ivp(
            fun=theta_eom,
            t_span=(t_prev, t_i),
            y0=y0_theta,
            t_eval=[t_i],
            rtol=reltol,
            atol=abstol,
            method=method,
        )
        theta_k_km1 = sol_theta.y[9:, -1].reshape(n, q)

        Phi_tk_t0 = Phi_k_km1 @ Phi_tk_t0
        theta_tk_t0 = Phi_k_km1 @ theta_tk_t0 + theta_k_km1
        Psi_k = np.block([[Phi_tk_t0, theta_tk_t0], [np.zeros((q, n)), np.eye(q)]])

        xbar_k = Phi_k_km1 @ xhat_prev
        Pbar_xk = Phi_k_km1 @ P_xk_prev @ Phi_k_km1.T
        Sbar_k = Phi_k_km1 @ S_prev + theta_k_km1
        Pbar_ck = Pbar_xk + Sbar_k @ P_cc0 @ Sbar_k.T
        Pbar_xck = Sbar_k @ P_cc0

        y_meas_k = []
        y_model_k = []
        Htilde_xk = []
        Htilde_ck = []
        stat_k = []

        for m in meas_k:
            st = station_map[m["station"]]
            stat_k.append(m["station"])
            y_meas_k.extend([float(m["rho_km"]), float(m["rho_dot_km_s"])])

            Rs, Vs, _ = st.ecef2eci(t_i, st.r_ecef, np.zeros(3))
            rho_vec = Xstar_k[:3] - Rs
            rho = np.linalg.norm(rho_vec)
            rho_dot = np.dot(rho_vec / rho, Xstar_k[3:] - Vs)
            y_model_k.extend([rho, rho_dot])

            Htilde_xk.append(H_range_rangerate(Xstar_k[:3], Xstar_k[3:], Rs, Vs))
            Htilde_ck.append(np.zeros((2, q)))

        y_k = np.asarray(y_meas_k) - np.asarray(y_model_k)
        Htilde_xk = np.vstack(Htilde_xk)
        Htilde_ck = np.vstack(Htilde_ck)
        R_k = R if len(meas_k) == 1 else la.block_diag(*([R] * len(meas_k)))

        K_k = Pbar_xk @ Htilde_xk.T @ np.linalg.solve(Htilde_xk @ Pbar_xk @ Htilde_xk.T + R_k, np.eye(y_k.size))
        Kc_k = Pbar_ck @ Htilde_xk.T @ np.linalg.solve(Htilde_xk @ Pbar_ck @ Htilde_xk.T + R_k, np.eye(y_k.size))

        xhat_k = xbar_k + K_k @ (y_k - Htilde_xk @ xbar_k)
        S_k = (I_n - K_k @ Htilde_xk) @ Sbar_k - K_k @ Htilde_ck
        xhat_ck = xbar_k + Kc_k @ (y_k - Htilde_xk @ xhat_k)

        A_k = I_n - K_k @ Htilde_xk
        P_xk = A_k @ Pbar_xk @ A_k.T + K_k @ R_k @ K_k.T
        P_ck = P_xk + S_k @ P_cc0 @ S_k.T
        P_xck = S_k @ P_cc0

        xEst.append(xhat_k)
        xcEst.append(xhat_ck)
        PEst.append(P_xk)
        PcEst.append(P_ck)
        PxcEst.append(P_xck)
        prefit_res.append(y_k)
        postfit_res.append(y_k - Htilde_xk @ xhat_k)
        postfit_res_c.append(y_k - Htilde_xk @ xhat_ck)
        XEst.append(Xstar_k + xhat_k)
        XcEst.append(Xstar_k + xhat_ck)
        Phi_total.append(Phi_tk_t0)
        Theta_total.append(theta_tk_t0)
        Psi.append(Psi_k)
        statVis.append(stat_k)

        iter_hist.append(
            {
                "xbar_k": xbar_k,
                "Pbar_xk": Pbar_xk,
                "Sbar_k": Sbar_k,
                "Pbar_ck": Pbar_ck,
                "Pbar_xck": Pbar_xck,
                "y_k": y_k,
                "y_model_k": np.asarray(y_model_k),
                "Htilde_xk": Htilde_xk,
                "Htilde_ck": Htilde_ck,
                "K_k": K_k,
                "Kc_k": Kc_k,
                "S_k": S_k,
                "Phi_k_km1": Phi_k_km1,
                "theta_k_km1": theta_k_km1,
                "t": t_i,
            }
        )

        t_prev = t_i
        Xstar_prev = Xstar_k
        xhat_prev = xhat_k
        P_xk_prev = P_xk
        S_prev = S_k

    return {
        "xEst": np.asarray(xEst),
        "xcEst": np.asarray(xcEst),
        "PEst": np.asarray(PEst),
        "PcEst": np.asarray(PcEst),
        "PxcEst": np.asarray(PxcEst),
        "prefit_res": prefit_res,
        "postfit_res": postfit_res,
        "postfit_res_c": postfit_res_c,
        "t_meas": np.asarray(t_epochs[1:num_meas]),
        "statVis": statVis,
        "XEst": np.asarray(XEst),
        "XcEst": np.asarray(XcEst),
        "Phi_total": np.asarray(Phi_total),
        "Theta_total": np.asarray(Theta_total),
        "Psi": Psi,
        "iter_hist": iter_hist,
    }
