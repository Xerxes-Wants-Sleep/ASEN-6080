import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import json

## This file has all the derived Jacobians and the affiliated orbit propagators


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


