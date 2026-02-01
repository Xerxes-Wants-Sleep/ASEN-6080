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