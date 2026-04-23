import numpy as np
from scipy.integrate import solve_ivp
from .srp_dyn_model import mu_sun_srp_stm_deriv


def calc_bplane(
    XPhi_3SOI: np.ndarray,
    t_3SOI: float,
    P_3SOI: np.ndarray,
    pConst,
    scConst,
    earth_state_func,
    sun_state_func,
    rtol: float = 1e-10,
    atol: float = 1e-10,
):
    n = 7

    X_3SOI = np.asarray(XPhi_3SOI[:n], dtype=float).reshape(n)
    P_3SOI = np.asarray(P_3SOI, dtype=float).reshape(n, n)

    r_sc = X_3SOI[0:3]
    v_sc = X_3SOI[3:6]

    r_earth, v_earth = earth_state_func(t_3SOI)

    # Earth-relative state for hyperbolic geometry
    r_vec = r_sc - r_earth
    v_vec = v_sc - v_earth

    r_mag = float(np.linalg.norm(r_vec))
    v_mag = float(np.linalg.norm(v_vec))
    mu = pConst.mu_earth

    # Hyperbolic orbit parameters
    e_vec = ((v_mag**2 - mu / r_mag) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))
    Phat = e_vec / e

    h_vec = np.cross(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))
    What = h_vec / h

    a = -mu / (v_mag**2 - 2.0 * mu / r_mag)
    Shat = v_vec / v_mag
    v_inf_hat = Shat

    Nhat = np.array([0.0, 0.0, 1.0])
    That = np.cross(Shat, Nhat)
    That_norm = np.linalg.norm(That)
    That = That / That_norm

    Rhat = np.cross(Shat, That)

    B_vec = r_vec - np.dot(r_vec, v_inf_hat) * v_inf_hat

    # DCM from STR to ECI
    STR2ECI = np.column_stack((Shat, That, Rhat))

    # True anomaly at 3SOI
    cNu = np.dot(r_vec / r_mag, Phat)

    # Hyperbolic anomaly
    arg = 1.0 + (v_mag**2 / mu) * ((a * (1.0 - e**2)) / (1.0 + e * cNu))
    f = np.arccosh(arg)

    # Linearized time of flight
    LTOF = (mu / v_mag**3) * (np.sinh(f) - f)

    # Integrate from 3SOI to B-plane crossing
    t_span = (t_3SOI, t_3SOI + LTOF)
    XPhi0 = np.hstack((X_3SOI, np.eye(n).reshape(-1)))

    sol = solve_ivp(
        fun=lambda t, y: mu_sun_srp_stm_deriv(
            t=t,
            XPhi=y,
            pConst=pConst,
            scConst=scConst,
            earth_state_func=earth_state_func,
            sun_state_func=sun_state_func,
        ),
        t_span=t_span,
        y0=XPhi0,
        rtol=rtol,
        atol=atol,
        method="RK45",
    )
    
    t_BPlane = sol.t
    XPhi_BPlane = sol.y.T

    X_crossing = XPhi_BPlane[-1, :n]
    Phi_crossing = XPhi_BPlane[-1, n:].reshape(n, n)

    # Covariance propagation
    P_Bplane = Phi_crossing @ P_3SOI @ Phi_crossing.T

    # Rotate covariance into STR coordinates
    ECI2STR = STR2ECI.T
    blkRot = np.eye(7)
    blkRot[0:3, 0:3] = ECI2STR
    blkRot[3:6, 3:6] = ECI2STR
    blkRot[6, 6] = 0.0

    P_Bplane = blkRot @ P_Bplane @ blkRot.T

    BdotR = float(np.dot(B_vec, Rhat))
    BdotT = float(np.dot(B_vec, That))

    sig_R = float(np.sqrt(P_Bplane[2, 2]))
    sig_T = float(np.sqrt(P_Bplane[1, 1]))
    sig_RT = float(P_Bplane[1, 2])

    return (
        BdotR,
        BdotT,
        sig_R,
        sig_T,
        sig_RT,
        X_crossing,
        P_Bplane,
        STR2ECI,
        XPhi_BPlane,
        t_BPlane,
    )
