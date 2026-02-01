import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("../../")

# ------------------------------------------------------------
# Problem 2a: Truth state differences (J2+J3) − (J2)
# 6 subplots: Δx,Δy,Δz,Δvx,Δvy,Δvz
# ------------------------------------------------------------

truth_j2_path = Path("../Homework1/prob2a_traj.csv")          # HW1 truth (J2)
truth_j3_path = Path("../Homework1/HW2_j3_on_truth.csv")      # new truth (J2+J3)
df_j2 = pd.read_csv(truth_j2_path)
df_j3 = pd.read_csv(truth_j3_path)

tcol = "t_s"
state_cols = ["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]

t = df_j2[tcol].to_numpy(float)
t2 = df_j3[tcol].to_numpy(float)

X_j2 = df_j2[state_cols].to_numpy(float)  # (N,6)
X_j3 = df_j3[state_cols].to_numpy(float)  # (N,6)

dX = X_j3 - X_j2  # (N,6)


labels = [
    r"$\Delta x$ (km)", r"$\Delta y$ (km)", r"$\Delta z$ (km)",
    r"$\Delta v_x$ (km/s)", r"$\Delta v_y$ (km/s)", r"$\Delta v_z$ (km/s)"
]

fig, axs = plt.subplots(3, 2, figsize=(10, 12), sharex=True)

axs = axs.reshape(3, 2)  # make absolutely sure it's 2D

k = 0
for i in range(3):
    for j in range(2):
        axs[i, j].plot(t, dX[:, k])
        axs[i, j].set_ylabel(labels[k])
        axs[i, j].grid(True)
        k += 1

# x-labels only on bottom row
axs[2, 0].set_xlabel("t (s)")
axs[2, 1].set_xlabel("t (s)")

fig.suptitle("Problem 2a: Truth State Differences (J2+J3) − (J2)", y=0.995)
fig.tight_layout()
plt.show()
