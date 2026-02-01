

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
J3 = -0.19686144647594
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

