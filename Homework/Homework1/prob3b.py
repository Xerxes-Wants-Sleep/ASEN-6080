import numpy as np
import sys
import json
import scipy as sp
import pandas as pd

sys.path.append("../../")


import numpy as np

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



with open("Solution_Checks/prob3b_solution.json", "r") as validation_file:
    my_json_obj = json.load(validation_file)

    r = np.array(my_json_obj["inputs"]["spacecraft_state"]["r"], dtype=np.float64)
    v = np.array(my_json_obj["inputs"]["spacecraft_state"]["v"], dtype=np.float64)
    Rs = np.array(my_json_obj["inputs"]["station_state"]["Rs"], dtype=np.float64)
    Vs = np.array(my_json_obj["inputs"]["station_state"]["Vs"], dtype=np.float64)
    H_expected = np.array(my_json_obj["outputs"]["Htilde"]["values"], dtype=np.float64)

##### CHECK #####

H_out1 = H_range_rangerate(r, v, Rs, Vs)

diff1 = H_out1 - H_expected

print("\n===== 3b =====")
print("H_out1:\n", H_out1)
print("\nH_expected:\n", H_expected)
print("\nMax abs diff:", np.max(np.abs(diff1)))
print("All close?", np.allclose(H_out1, H_expected, rtol=1e-12, atol=1e-12))



################# C #######################


def Htilde_obs_rho_rhod(r_sc, v_sc, Rs, Vs, eps=1e-12):
    """
    Returns 2x3 partials of [rho; rhodot] w.r.t. station position Rs.

    Uses the homework's implicit ground-station kinematics:
        Vs = omega x Rs   (omega along +z)
    so d(Vs)/d(Rs) contributes to d(rhodot)/d(Rs).

    rho   = ||r_sc - Rs||
    rhod  = ( (r_sc - Rs)·(v_sc - Vs) ) / rho
    """
    r_sc = np.asarray(r_sc, dtype=float).reshape(3,)
    v_sc = np.asarray(v_sc, dtype=float).reshape(3,)
    Rs   = np.asarray(Rs,   dtype=float).reshape(3,)
    Vs   = np.asarray(Vs,   dtype=float).reshape(3,)

    r = r_sc - Rs
    rho = np.linalg.norm(r)
    if rho < eps:
        raise ValueError("Range rho is ~0; LOS undefined.")

    rho_hat = r / rho

    # relative velocity
    v = v_sc - Vs
    rv = np.dot(r, v)

    # --- infer omega_z from Vs = omega x Rs with omega = [0,0,omega_z] ---
    # Vs = [-omega_z*Rs_y, omega_z*Rs_x, 0]
    denom = Rs[0]**2 + Rs[1]**2
    if denom < eps:
        raise ValueError("Cannot infer omega_z when Rs_x and Rs_y are ~0.")

    omega_z = (Rs[0]*Vs[1] - Rs[1]*Vs[0]) / denom
    omega = np.array([0.0, 0.0, omega_z])

    # d(rho)/d(Rs) = -r^T/rho = -rho_hat^T
    drho_dRs = (-rho_hat).reshape(1, 3)

    # Base term (holding Vs fixed):  - d(rhod)/dR
    base = -(v / rho - (rv / rho**3) * r)

    # Extra chain-rule term due to Vs(Rs):  (omega x r)/rho
    extra = np.cross(omega, r) / rho

    drhod_dRs = (base + extra).reshape(1, 3)

    return np.vstack((drho_dRs, drhod_dRs))



with open("Solution_Checks/prob3d_solution.json", "r") as validation_file:
    my_json_obj = json.load(validation_file)

    r  = np.array(my_json_obj["inputs"]["spacecraft_state"]["r"], dtype=np.float64)
    v  = np.array(my_json_obj["inputs"]["spacecraft_state"]["v"], dtype=np.float64)
    Rs = np.array(my_json_obj["inputs"]["station_state"]["Rs"], dtype=np.float64)
    Vs = np.array(my_json_obj["inputs"]["station_state"]["Vs"], dtype=np.float64)
    H_expected = np.array(my_json_obj["outputs"]["Htilde"]["values"], dtype=np.float64)

H_out = Htilde_obs_rho_rhod(r, v, Rs, Vs)

print("H_out:\n", H_out)
print("\nH_expected:\n", H_expected)

diff = H_out - H_expected
print("\nMax abs diff:", np.max(np.abs(diff)))
print("All close?", np.allclose(H_out, H_expected, rtol=1e-12, atol=1e-12))