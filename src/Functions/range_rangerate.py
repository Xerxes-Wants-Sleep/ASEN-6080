import numpy as np

'''This is the range and range rate sensitivity to changes in state'''


def H_range_rangerate(R, V, Rs, Vs, eps=1e-12):
    """
    Measurement partials for simplified range and range-rate:
        rho  = ||R - Rs||
        rhod = (R - Rs)·(V - Vs) / rho

    Inputs:  R,V,Rs,Vs are length-3 arrays (or 3x1 vectors)
    Output:  H is 2x6 Jacobian wrt X = [R; V]
             rows: [rho, rhod], cols: [R (3), V (3)]
    """
    R  = np.asarray(R,  dtype=float).reshape(3,)
    V  = np.asarray(V,  dtype=float).reshape(3,)
    Rs = np.asarray(Rs, dtype=float).reshape(3,)
    Vs = np.asarray(Vs, dtype=float).reshape(3,)

    r = R - Rs
    v = V - Vs

    rho = np.linalg.norm(r)
    rho_hat = r / rho    
    rhod = np.dot(r, v) / rho

    # Partials
    drho_dR = rho_hat.reshape(1, 3)      # 1x3
    drho_dV = np.zeros((1, 3))           # 1x3

    # d(rhod)/dR
    drhod_dR = (v / rho - (np.dot(r, v) / rho**3) * r).reshape(1, 3)

    # d(rhod)/dV
    drhod_dV = rho_hat.reshape(1, 3)

    H = np.block([
        [drho_dR,  drho_dV],
        [drhod_dR, drhod_dV]])  # 2x6

    return H




def H_range_range_rate_wrt_station(r_sc, v_sc, Rs, Vs, eps=1e-12):
    """
    Returns 2x3 partials of [rho; rhodot] w.r.t. station position Rs only.

    rho   = ||r_sc - Rs||
    rhod  = ( (r_sc - Rs)·(v_sc - Vs) ) / rho
    """
    r_sc = np.asarray(r_sc, dtype=float).reshape(3,)
    v_sc = np.asarray(v_sc, dtype=float).reshape(3,)
    Rs   = np.asarray(Rs,   dtype=float).reshape(3,)
    Vs   = np.asarray(Vs,   dtype=float).reshape(3,)

    r = r_sc - Rs
    v = v_sc - Vs

    rho = np.linalg.norm(r)
    rho_hat = r / rho
    rv = np.dot(r, v)
    drho_dRs = (-rho_hat).reshape(1, 3)
    drhod_dRs = (-(v / rho - (rv / rho**3) * r)).reshape(1, 3)

    H = np.vstack((drho_dRs, drhod_dRs))
    return H



##### Project 1 #####

def H_tilde_range_rangerate_augmented(
    state: np.ndarray,
    stat_idx: int,
    *,
    meas_include=(True, True),
    omega_rad_s: float = 7.2921158553e-5,
    station_start_index: int = 9,
    num_stations: int = 3,
    return_pred: bool = False,
    eps: float = 1e-12,
):
    """
    Full-state measurement Jacobian for range & range-rate with station positions in the state.

    Assumes state layout:
      [r(3), v(3), (any params...), stations positions ...]
    where stations begin at station_start_index and each station contributes 3 states.

    Station inertial velocity model:
      Vs = omega x Rs

    Parameters
    ----------
    state : (n,) ndarray
    stat_idx : int
        Visible station index in [0..num_stations-1]
    meas_include : (bool,bool)
        (include_range, include_rangerate)
    omega_rad_s : float
        Rotation rate for Vs = omega x Rs
    station_start_index : int
        Index in state where station positions begin
    num_stations : int
        Number of stations in the state
    return_pred : bool
        If True, also return predicted yhat = [rho, rho_dot] (or subset)
    """
    x = np.asarray(state, dtype=float).reshape(-1)
    n = x.size

    include_rho, include_rhod = meas_include
    # --- spacecraft state ---
    R = x[0:3]
    V = x[3:6]

    # --- visible station position from state ---
    base = station_start_index + 3 * stat_idx
    if base + 3 > n:
        raise ValueError("Station slice exceeds state dimension; check station_start_index/num_stations/state size.")

    Rs = x[base:base+3]

    # --- station velocity model ---
    omega = np.array([0.0, 0.0, float(omega_rad_s)], dtype=float)
    Vs = np.cross(omega, Rs)

    # --- predicted measurement ---
    rho_vec = R - Rs
    rho = float(np.linalg.norm(rho_vec))
    if rho < eps:
        raise ValueError("Range too small; check geometry/units.")
    rho_hat = rho_vec / rho
    v_rel = V - Vs
    rho_dot = float(np.dot(rho_hat, v_rel))

    # --- spacecraft partials (2x6) using your existing primitive ---
    H_sc = H_range_rangerate(R, V, Rs, Vs, eps=eps)  # 2x6

    # --- station position partials (2x3) ---
    # Start with your existing station-pos partial that treats Vs independent:
    H_Rs = H_range_range_rate_wrt_station(R, V, Rs, Vs, eps=eps)  # 2x3

    # Add coupling term because Vs = omega x Rs:
    # d(rho_dot)/d(Vs) = -rho_hat^T
    # d(Vs)/d(Rs) = [omega]_x  (cross-product matrix)
    w_tilde = np.array([[0.0, -omega_rad_s, 0.0],
                        [omega_rad_s, 0.0, 0.0],
                        [0.0, 0.0, 0.0]], dtype=float)
    H_Rs[1:2, :] += (-rho_hat.reshape(1, 3)) @ w_tilde  # only range-rate row

    # --- build full H (2 x n), only visible station block nonzero ---
    H_full = np.zeros((2, n), dtype=float)
    H_full[:, 0:6] = H_sc
    H_full[:, base:base+3] = H_Rs

    # --- select rows ---
    rows = []
    if include_rho:
        rows.append(0)
    if include_rhod:
        rows.append(1)
    H_out = H_full[rows, :]

    if not return_pred:
        return H_out

    yhat_full = np.array([rho, rho_dot], dtype=float)
    return yhat_full[rows], H_out