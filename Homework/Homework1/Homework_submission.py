import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import json

## I will make a git for next homework to make this all more legible

def accel_wJ2J3(r, mu, J2, J3, Re=6378, j2=True, j3=True):
    """
    Acceleration a(r) = a_mu + a_J2 + a_J3 in Cartesian
    j2 j3 = togglable
    """ 

    x, y, z = r
    r2 = x*x + y*y + z*z
    rmag = np.sqrt(r2)

    # Central term
    a = -mu * r / (rmag**3)

    # J2 
    if j2:
        z2 = z*z
        u2 = np.array([
            x * (r2 - 5.0*z2),
            y * (r2 - 5.0*z2),
            z * (3.0*r2 - 5.0*z2)
        ])
        a -= (3.0*mu*J2*(Re**2) / (2.0*(rmag**7))) * u2

    # J3
    if j3:
        z2 = z*z
        r4 = r2*r2
        z4 = z2*z2
        C = 7.0*z2 - 3.0*r2
        D = 3.0*r4 - 30.0*r2*z2 + 35.0*z4
        w = np.array([
            5.0*x*z*C,
            5.0*y*z*C,
            D
        ])
        a += (mu*J3*(Re**3) / (2.0*(rmag**9))) * w

    return a


def dadr_wJ2J3(r, mu, J2, J3, Re=6378, j2=True, j3=True):
    """
    Gravity-gradient matrix G = da/dr (3x3) for a = a_mu + a_J2 + a_J3.

    Returns
    G : array, shape (3,3)
    """
    x, y, z = r
    r2 = x*x + y*y + z*z
    rmag = np.sqrt(r2)
    I = np.eye(3)
    rrT = np.outer(r, r)

    # Central term
    G = (mu / (rmag**5)) * (3.0*rrT - r2*I)

    # J2 
    if j2:
        z2 = z*z
        u2 = np.array([
            x * (r2 - 5.0*z2),
            y * (r2 - 5.0*z2),
            z * (3.0*r2 - 5.0*z2)
        ])

        
        M2 = np.array([
            [(r2 - 5.0*z2) + 2.0*x*x,  2.0*x*y,              -8.0*x*z],
            [2.0*x*y,                  (r2 - 5.0*z2) + 2.0*y*y, -8.0*y*z],
            [6.0*x*z,                  6.0*y*z,              (3.0*r2 - 5.0*z2) - 4.0*z2]
        ])

        coeff = (3.0*mu*J2*(Re**2)) / (2.0*(rmag**7))
        G -= coeff * (M2 - (7.0/r2)*np.outer(u2, r))

    # J3
    if j3:
        z2 = z*z
        r4 = r2*r2
        z4 = z2*z2

        C = 7.0*z2 - 3.0*r2
        D = 3.0*r4 - 30.0*r2*z2 + 35.0*z4

        w = np.array([
            5.0*x*z*C,
            5.0*y*z*C,
            D
        ])

        M3 = np.array([
            [5.0*z*(C - 6.0*x*x),   -30.0*x*y*z,        5.0*x*(C + 8.0*z2)],
            [-30.0*x*y*z,           5.0*z*(C - 6.0*y*y), 5.0*y*(C + 8.0*z2)],
            [12.0*x*(r2 - 5.0*z2),  12.0*y*(r2 - 5.0*z2), 16.0*z*(5.0*z2 - 3.0*r2)]
        ])

        coeff = (mu*J3*(Re**3)) / (2.0*(rmag**9))
        G += coeff * (M3 - (9.0/r2)*np.outer(w, r))

    return G


def da_dparams_wJ2J3(r, mu, J2, J3, Re=6378, j2=True, j3=True):
    """
    Parameter partials: da WRT mu, j2, j3.
    """
    a = accel_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)
    
    da_dmu = a / mu

    # da/dj2
    if j2:
        x, y, z = r
        r2 = x*x + y*y + z*z
        rmag = np.sqrt(r2)
        z2 = z*z
        u2 = np.array([
            x * (r2 - 5.0*z2),
            y * (r2 - 5.0*z2),
            z * (3.0*r2 - 5.0*z2)
        ])
        da_dJ2 = -(3.0*mu*(Re**2) / (2.0*(rmag**7))) * u2
    else:
        da_dJ2 = np.zeros(3)

    # da/dj3
    if j3:
        x, y, z = r
        r2 = x*x + y*y + z*z
        rmag = np.sqrt(r2)
        z2 = z*z
        r4 = r2*r2
        z4 = z2*z2
        C = 7.0*z2 - 3.0*r2
        D = 3.0*r4 - 30.0*r2*z2 + 35.0*z4
        w = np.array([5.0*x*z*C, 5.0*y*z*C, D])
        da_dJ3 = (mu*(Re**3) / (2.0*(rmag**9))) * w
    else:
        da_dJ3 = np.zeros(3)

    return da_dmu, da_dJ2, da_dJ3


def state_builder(state9, Re=6378, j2=True, j3=True):
    """
    Build the 9x9 A-matrix for augmented state:
      X = [r(3), v(3), mu, J2, J3].

    Returns
    -------
    A : 9x9 array of STM
    """
    state9 = np.asarray(state9, dtype=float).reshape(-1)

    r = state9[0:3]
    v = state9[3:6]  # not needed for Jacobian 
    mu = state9[6]
    J2 = state9[7]
    J3 = state9[8]

    G = dadr_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)         # 3x3
    da_dmu, da_dJ2, da_dJ3 = da_dparams_wJ2J3(r, mu, J2, J3, Re=Re, j2=j2, j3=j3)

    A = np.zeros((9, 9), dtype=float)

    # dr/dt = v
    A[0:3, 3:6] = np.eye(3)

    # dv/dt = a
    A[3:6, 0:3] = G
    # dv/dt partials wrt params
    A[3:6, 6] = da_dmu
    A[3:6, 7] = da_dJ2
    A[3:6, 8] = da_dJ3
    return A


## Validation...please work

with open('Solution_Checks/prob1c_solution.json', "r") as validation_file:
    my_json_obj = json.load(validation_file)

    r  = np.array(my_json_obj["inputs"]["state"]["r"], dtype=float)
    v  = np.array(my_json_obj["inputs"]["state"]["v"], dtype=float)
    mu = float(my_json_obj["inputs"]["state"]["mu"])
    J2 = float(my_json_obj["inputs"]["state"]["J2"])
    J3 = float(my_json_obj["inputs"]["state"]["J3"])

state9 = np.hstack((r, v, mu, J2, J3))

A_calc = state_builder(state9, Re=6378, j2=True, j3=True)

# print(A_calc)

A_ref  = np.array(my_json_obj["outputs"]["A_matrix"]["values"], dtype=float)
# print(A_ref)

check = A_calc - A_ref

df_check = pd.DataFrame(check)
print("This is the check (A_calc - A_ref):")
print(df_check.to_string(index=False, header=False,
                         float_format=lambda x: f"{x: .8e}"))





### 2a


import numpy as np
import sys
import json
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../../")

from src.Functions.jacobians import *
from src.Functions.kep2cart import Keplarian_to_Cartesian

with open('prob2_OE.json', "r") as validation_file:
    my_json_obj = json.load(validation_file)

    a = np.float64(my_json_obj["orbit_elements"]["a_km"])
    e = np.float64(my_json_obj["orbit_elements"]["e"])
    i_deg = np.float64(my_json_obj["orbit_elements"]["i_deg"])
    raan_deg = np.float64(my_json_obj["orbit_elements"]["raan_deg"])
    w_deg = np.float64(my_json_obj["orbit_elements"]["argp_deg"])
    ta_deg = np.float64(my_json_obj["orbit_elements"]["ta_deg"])

mu: np.float64 = 398600.4415
Re: np.integer = 6378
J2: np.float64 = 0.0010826269
tf: np.float64 = 15 * 2 * np.pi * np.sqrt((a**3/mu))

r_N, v_N, _ = Keplarian_to_Cartesian(mu, a, e, i_deg, raan_deg, w_deg, ta_deg)

X0 = np.array(np.hstack((r_N, v_N, mu, J2, 0)))

########### Solution Check ############

truth = pd.read_csv(
    "Solution_Checks/HW1_truth.txt",
    sep=r"\s+",
    header=None,
    names=["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"],
    engine="python"
)

t_truth = truth["t_s"].to_numpy()
X_truth = truth[["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]].to_numpy()



# Problem 2a
fun = lambda t, X: orbit_propagator_aug9(t, X, Re=Re, j2=True, j3=False)
sol = sp.integrate.solve_ivp(fun, (t_truth[0], t_truth[-1]), X0, t_eval=t_truth, rtol=1e-10, atol=1e-10, method="DOP853")


X_model = sol.y[:6, :].T
cols = ["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s", "mu", "J2", "J3"]
data = np.column_stack((sol.t, sol.y.T))
df = pd.DataFrame(data, columns=cols)

# write CSV
df.to_csv("prob2a_traj.csv", index=False)
final_stm = sol.y[-1,:]


err = X_model - X_truth             # Nx6
pos_err = err[:, :3]                # Nx3
vel_err = err[:, 3:]                # Nx3

pos_err_norm = np.linalg.norm(pos_err, axis=1)
vel_err_norm = np.linalg.norm(vel_err, axis=1)

print("Max |pos error| (km):", pos_err_norm.max())
print("Max |vel error| (km/s):", vel_err_norm.max())



plt.figure()
plt.scatter(t_truth, pos_err_norm)
plt.xlabel("t (s)")
plt.ylabel("||position error|| (km)")
plt.grid(True)

plt.figure()
plt.scatter(t_truth, vel_err_norm)
plt.xlabel("t (s)")
plt.ylabel("||velocity error|| (km/s)")
plt.grid(True)

# plt.show()

# ---------------- 2a: Overlay truth vs model (6 subplots) ----------------


labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]

fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
axs = axs.ravel()

for k in range(6):
    axs[k].scatter(t_truth, X_truth[:, k], label="Truth", s=8, alpha=0.8)
    axs[k].scatter(t_truth, X_model[:, k], label="Calculated Traj", s=8, alpha=0.6)
    axs[k].set_ylabel(labels[k])
    axs[k].grid(True)

axs[4].set_xlabel("t (s)")
axs[5].set_xlabel("t (s)")

axs[0].legend(loc="best")
fig.suptitle("2a: Truth Trajectory vs Numerical Propagation")
fig.tight_layout()
plt.show()



#### 2b



import numpy as np
import sys
import json
import scipy as sp
import pandas as pd
sys.path.append("../../")
from src.Functions.jacobians import *
from src.Functions.kep2cart import Keplarian_to_Cartesian


################# Problem 2b ########################


mu: np.float64 = 398600.4415
Re: np.integer = 6378
J2: np.float64 = 1.08262668e-3

with open('Solution_Checks/prob2b_solution.json', "r") as validation_file2:
    my_json_obj2 = json.load(validation_file2)

    X0 = np.array(my_json_obj2["inputs"]["X0"]["values"], dtype=float)
    Phi0 = np.array(my_json_obj2["inputs"]["Phi0"]["values"], dtype=float)
    X_out = np.array(my_json_obj2["outputs"]["Xdot"]["values"], dtype=float)
    phi_out = np.array(my_json_obj2["outputs"]["Phidot"]["values"], dtype=float)


t = 0
state9 = np.array([*X0[0:6], mu, X0[6], 0])  #tuple unpacking
remove = np.array([6, 8])

state_stuff = stm(t, state9, Phi0, remove, Re=6378, j2=True, j3=True)

state_check = state_stuff[0:7]
phi_check = state_stuff[9:]
phi_check_matrix_residuals = phi_check.reshape(7, 7) - phi_out
phi_check_matrix = phi_check.reshape(7, 7)

# print(state_check-X_out)

df_phi = pd.DataFrame(phi_check_matrix_residuals)
print("Phi Difference", df_phi.to_string(index=False, header=False, float_format=lambda x: f"{x: .8e}"))

# df_phi = pd.DataFrame(phi_check_matrix)
# print(df_phi.to_string(index=False, header=False, float_format=lambda x: f"{x: .8e}"))

df_phi = pd.DataFrame(state_check-X_out)
print("State Difference", df_phi.to_string(index=False, header=False, float_format=lambda x: f"{x: .8e}"))



################# Problem 2c ########################

import numpy as np
import sys
import json
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")
from src.Functions.jacobians import *
from src.Functions.kep2cart import Keplarian_to_Cartesian


################# Problem 2c ########################


############### Data Load In ####################

with open('prob2_OE.json', "r") as validation_file:
    my_json_obj = json.load(validation_file)

    a = np.float64(my_json_obj["orbit_elements"]["a_km"])
    e = np.float64(my_json_obj["orbit_elements"]["e"])
    i_deg = np.float64(my_json_obj["orbit_elements"]["i_deg"])
    raan_deg = np.float64(my_json_obj["orbit_elements"]["raan_deg"])
    w_deg = np.float64(my_json_obj["orbit_elements"]["argp_deg"])
    ta_deg = np.float64(my_json_obj["orbit_elements"]["ta_deg"])

    x = np.float64(my_json_obj["pert"]["x"])
    y = np.float64(my_json_obj["pert"]["y"])
    z = np.float64(my_json_obj["pert"]["z"])
    xdot = np.float64(my_json_obj["pert"]["dx"])
    ydot = np.float64(my_json_obj["pert"]["dy"])
    zdot = np.float64(my_json_obj["pert"]["dz"])


truth = pd.read_csv("prob2a_traj.csv")

##################### Problem ######################

t_truth = truth["t_s"].to_numpy(dtype=float)

mu: np.float64 = 398600.4415
Re: np.integer = 6378
J2: np.float64 = 0.0010826269
# tf: np.float64 = 15 * 2 * np.pi * np.sqrt((a**3/mu))

r_N, v_N, _ = Keplarian_to_Cartesian(mu, a, e, i_deg, raan_deg, w_deg, ta_deg)
X0_pure = np.array(np.hstack((r_N, v_N, mu, J2, 0)))
X_pert = np.array(np.hstack((x, y, z, xdot, ydot, zdot, 0, 0, 0)))

X0 = X0_pure + X_pert
# Perturbed 2a 

fun = lambda t, X: orbit_propagator_aug9(t, X, Re=Re, j2=True, j3=False)
sol = sp.integrate.solve_ivp(fun, (t_truth[0], t_truth[-1]), X0, t_eval=t_truth, rtol=1e-10, atol=1e-10, method="DOP853")


X_model = sol.y[:6, :].T
cols = ["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s", "mu", "J2", "J3"]
data = np.column_stack((sol.t, sol.y.T))
df = pd.DataFrame(data, columns=cols)

# write CSV
df.to_csv("prob2c_traj_pert.csv", index=False)






X_ref = truth[["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]].to_numpy(dtype=float)
X_pert_traj = sol.y[:6, :].T

dx_nonlin = X_pert_traj - X_ref 



####################### Pert Linearized State ##########################

remove = np.array([6, 8], dtype=int)
Phi0 = np.eye(7)
dx0 = np.array([x, y, z, xdot, ydot, zdot], dtype=float)
dx0_7 = np.hstack((dx0, 0.0))
y0 = np.hstack((X0_pure, Phi0.flatten()))

fun_stm = lambda t, y: stm(t, state9=y[:9], phi=y[9:].reshape(7, 7), rows_col_to_remove=remove, Re=Re, j2=True, j3=False)
sol_stm = sp.integrate.solve_ivp(fun_stm, (t_truth[0], t_truth[-1]), y0, t_eval=t_truth, rtol=1e-10, atol=1e-10, method="DOP853")

Phi_hist = sol_stm.y[9:, :].T.reshape(-1, 7, 7)

N = len(t_truth)
dx_stm_7 = np.zeros((N, 7))

for i in range(N):
    Phi_i = Phi_hist[i, :, :]
    dx_stm_7[i, :] = Phi_i @ dx0_7

dx_stm = dx_stm_7[:, :6]



# ---------------- Save delta-x histories to CSV ----------------
cols_dx = ["t_s", "dx_km", "dy_km", "dz_km", "dvx_km_s", "dvy_km_s", "dvz_km_s"]

# Nonlinear (truth) delta-x
df_dx_nonlin = pd.DataFrame(
    np.column_stack((t_truth, dx_nonlin)),
    columns=cols_dx
)
df_dx_nonlin.to_csv("prob2c_dx_nonlin.csv", index=False)

# STM-predicted delta-x
df_dx_stm = pd.DataFrame(
    np.column_stack((t_truth, dx_stm)),
    columns=cols_dx
)
df_dx_stm.to_csv("prob2c_dx_stm.csv", index=False)


# print("X0_pure:", X0_pure)
# print("X_pert :", X_pert)
# print("X0     :", X0)
# print("mu,J2,J3 in X0:", X0[6], X0[7], X0[8])




# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

labels = ["δx (km)", "δy (km)", "δz (km)", "δvx (km/s)", "δvy (km/s)", "δvz (km/s)"]

fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
axs = axs.ravel()

for k in range(6):
    axs[k].scatter(t_truth, dx_nonlin[:, k], label="Nonlinear truth", s=2, alpha=0.7)
    axs[k].scatter(t_truth, dx_stm[:, k],   label="STM prediction",  s=2, alpha=0.5)
    axs[k].set_ylabel(labels[k])
    axs[k].grid(True)

axs[4].set_xlabel("t (s)")
axs[5].set_xlabel("t (s)")

axs[0].legend(loc="best")
fig.suptitle("2c: Deviation Components vs Time (Nonlinear vs STM)")
fig.tight_layout()



## d
d_dx = dx_nonlin - dx_stm

labels_d = ["Δδx (km)", "Δδy (km)", "Δδz (km)", "Δδvx (km/s)", "Δδvy (km/s)", "Δδvz (km/s)"]

fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
axs = axs.ravel()

for k in range(6):
    axs[k].scatter(t_truth, d_dx[:, k], s=2, alpha=0.7)
    axs[k].set_ylabel(labels_d[k])
    axs[k].grid(True)

axs[4].set_xlabel("t (s)")
axs[5].set_xlabel("t (s)")

fig.suptitle("2d: Δδx(t) = δx_nonlin(t) − δx_STM(t)")
fig.tight_layout()
plt.show()




## Prob 3b

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


## Prob 4a


import numpy as np
import sys
import json
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")

from src.Functions.stations import Stations

truth = pd.read_csv("prob2a_traj.csv")
t = truth["t_s"].to_numpy(float)
r = truth[["x_km", "y_km", "z_km"]].to_numpy(float)
v = truth[["vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)

# Stations (calling class)
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]

# Collect all measurements in one list
all_meas = []

for k in range(len(t)):
    for st in stations:
        m = st.measure(r[k], v[k], float(t[k]))
        if m is not None:
            all_meas.append(m)

# First/last measurement times

t_first = min(m["t"] for m in all_meas)
t_last  = max(m["t"] for m in all_meas)
print("First measurement time (s):", t_first)
print("Last measurement time (s):", t_last)






# Plot by station: Range
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue
    tt  = np.array([m["t"] for m in st_meas], dtype=float)
    rho = np.array([m["rho_km"] for m in st_meas], dtype=float)
    plt.scatter(tt, rho, s=6, alpha=0.8, label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Range ρ (km)")
plt.title("Range Measurements)")
plt.grid(True)
plt.legend()


# Plot by station: Range-rate
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue
    tt   = np.array([m["t"] for m in st_meas], dtype=float)
    rhod = np.array([m["rho_dot_km_s"] for m in st_meas], dtype=float)
    plt.scatter(tt, rhod, s=6, alpha=0.8, label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Range-rate ρ̇ (km/s)")
plt.title("Range-rate measurements")
plt.grid(True)
plt.legend()


# Plot by station: Elevation
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue
    tt   = np.array([m["t"] for m in st_meas], dtype=float)
    elev = np.array([np.rad2deg(m["elev_rad"]) for m in st_meas], dtype=float)
    plt.scatter(tt, elev, s=6, alpha=0.8, label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Elevation (deg)")
plt.title("Elevation Angles at Measurement Times")
plt.grid(True)
plt.legend()

plt.show()





################### D ############################

def add_gaussian_noise(y: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return y + rng.normal(loc=0.0, scale=sigma, size=y.shape)


sigma_km_s = 5e-7  # 0.5 mm/s in km/s

# Original vs noisy
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue

    tt = np.array([m["t"] for m in st_meas], dtype=float)
    rhod = np.array([m["rho_dot_km_s"] for m in st_meas], dtype=float)
    rhod_noisy = add_gaussian_noise(rhod, sigma_km_s, seed=123)

    plt.scatter(tt, rhod,       s=6, alpha=0.8, label=f"{st.name} original")
    plt.scatter(tt, rhod_noisy, s=6, alpha=0.6, label=f"{st.name} noisy")

plt.xlabel("t (s)")
plt.ylabel("Range-rate ρ̇ (km/s)")
plt.title("Range-rate: Original vs Noisy (σ = 0.5 mm/s)")
plt.grid(True)
plt.legend()


# Residual plot
plt.figure()
for st in stations:
    st_meas = [m for m in all_meas if m["station"] == st.name]
    if not st_meas:
        continue

    tt = np.array([m["t"] for m in st_meas], dtype=float)
    rhod = np.array([m["rho_dot_km_s"] for m in st_meas], dtype=float)
    rhod_noisy = add_gaussian_noise(rhod, sigma_km_s, seed=123)
    residual = rhod_noisy - rhod

    plt.scatter(tt, residual, s=6, alpha=0.8, label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Noisy − original (km/s)")
plt.title("Range-Rate Noise Residuals")
plt.grid(True)
plt.legend()

plt.show()


# Prob 4c

import numpy as np
import sys
import json
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")

from src.Functions.stations import Stations





def RUdoppler( station: Stations, r_sc_eci: np.ndarray, v_sc_eci: np.ndarray, t: float, f_tr_ref_hz: float = 8.44e9, c_km_s: float = 2.99792458e5) -> dict:
    elev = station.elevation(t, r_sc_eci)

    if elev < station.elevation_mask_rad:
        return {"station": station.name, "t": t, "RU": np.nan, "f_shift_hz": np.nan, "elev_rad": elev}

    m = station.measure(r_sc_eci, v_sc_eci, t)
    rho_km = float(m["rho_km"])
    rhodot_km_s = float(m["rho_dot_km_s"])

    f_shift = -2.0 * rhodot_km_s * f_tr_ref_hz / c_km_s
    RU = (221.0 / 749.0) * (rho_km / c_km_s) * f_tr_ref_hz

    return {"station": station.name, "t": t, "RU": RU, "f_shift_hz": f_shift, "elev_rad": elev}



truth = pd.read_csv("prob2a_traj.csv")
t = truth["t_s"].to_numpy(float)
r = truth[["x_km", "y_km", "z_km"]].to_numpy(float)
v = truth[["vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)

stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]

all_dsn = []

for k in range(len(t)):
    for st in stations:
        y = RUdoppler(st, r[k], v[k], float(t[k]))  # returns dict
        all_dsn.append(y)


plt.figure()
for st in stations:
    st_dsn = [d for d in all_dsn if d["station"] == st.name and np.isfinite(d["RU"])]
    if not st_dsn:
        continue
    plt.plot([d["t"] for d in st_dsn],
             [d["RU"] for d in st_dsn],
             marker=".", linestyle="None", label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Range Units (RU)")
plt.title("2-way Range in Range Units (RU)")
plt.grid(True)
plt.legend()

plt.figure()
for st in stations:
    st_dsn = [d for d in all_dsn if d["station"] == st.name and np.isfinite(d["f_shift_hz"])]
    if not st_dsn:
        continue
    plt.plot([d["t"] for d in st_dsn],
             [d["f_shift_hz"] for d in st_dsn],
             marker=".", linestyle="None", label=st.name)

plt.xlabel("t (s)")
plt.ylabel("Doppler shift (Hz)")
plt.title("2-way Doppler shift (Hz)")
plt.grid(True)
plt.legend()
plt.show()


##Prob 4d

import numpy as np
import sys
import json
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")

from src.Functions.stations import Stations

truth = pd.read_csv("prob2a_traj.csv")
t = truth["t_s"].to_numpy(float)
r = truth[["x_km", "y_km", "z_km"]].to_numpy(float)
v = truth[["vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)

# Stations (calling class)
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]






