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
J3 = -2.5324e-6
tf: np.float64 = 15 * 2 * np.pi * np.sqrt((a**3/mu))

r_N, v_N, _ = Keplarian_to_Cartesian(mu, a, e, i_deg, raan_deg, w_deg, ta_deg)

X0 = np.array(np.hstack((r_N, v_N, mu, J2, J3)))

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
fun = lambda t, X: orbit_propagator_aug9(t, X, Re=Re, j2=True, j3=True)
sol = sp.integrate.solve_ivp(fun, (t_truth[0], t_truth[-1]), X0, t_eval=t_truth, rtol=1e-10, atol=1e-10, method="DOP853")


X_model = sol.y[:6, :].T
cols = ["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s", "mu", "J2", "J3"]
data = np.column_stack((sol.t, sol.y.T))
df = pd.DataFrame(data, columns=cols)

# write CSV
df.to_csv("HW2_j3_on_truth.csv", index=False)
final_stm = sol.y[-1,:]


# err = X_model - X_truth             # Nx6
# pos_err = err[:, :3]                # Nx3
# vel_err = err[:, 3:]                # Nx3

# pos_err_norm = np.linalg.norm(pos_err, axis=1)
# vel_err_norm = np.linalg.norm(vel_err, axis=1)

# print("Max |pos error| (km):", pos_err_norm.max())
# print("Max |vel error| (km/s):", vel_err_norm.max())



# plt.figure()
# plt.scatter(t_truth, pos_err_norm)
# plt.xlabel("t (s)")
# plt.ylabel("||position error|| (km)")
# plt.grid(True)

# plt.figure()
# plt.scatter(t_truth, vel_err_norm)
# plt.xlabel("t (s)")
# plt.ylabel("||velocity error|| (km/s)")
# plt.grid(True)

# # plt.show()

# # ---------------- 2a: Overlay truth vs model (6 subplots) ----------------


# labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]

# fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
# axs = axs.ravel()

# for k in range(6):
#     axs[k].scatter(t_truth, X_truth[:, k], label="Truth", s=8, alpha=0.8)
#     axs[k].scatter(t_truth, X_model[:, k], label="Calculated Traj", s=8, alpha=0.6)
#     axs[k].set_ylabel(labels[k])
#     axs[k].grid(True)

# axs[4].set_xlabel("t (s)")
# axs[5].set_xlabel("t (s)")

# axs[0].legend(loc="best")
# fig.suptitle("2a: Truth Trajectory vs Numerical Propagation")
# fig.tight_layout()
# plt.show()

