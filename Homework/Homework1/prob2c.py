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

t_truth = truth["t_s"].to_numpy(dtype=np.ndarray)

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
