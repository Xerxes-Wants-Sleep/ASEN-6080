import numpy as np


def state_noise_compensation(delta_t, n, m, Q):
    """Build discrete process noise from continuous white acceleration noise.

    delta_t: time step (s)
    n: state dimension (6 for [r, v])
    m: process-noise dimension (3 for acceleration noise)
    Q: continuous acceleration covariance (m x m)
    """

    dt = float(delta_t)
    gamma = np.zeros((n, m), dtype=float)
    gamma[0:3, 0:3] = dt**2 / 2 * np.eye(3)
    gamma[3:6, 0:3] = dt * np.eye(3)

    snc_eci = gamma @ Q @ gamma.T
    return snc_eci


def eci_to_ric_rotation(r_eci, v_eci):
    """
    Build the ECI->RIC rotation matrix using the repo's RSW/RIC convention:
      Rhat = r / ||r||
      What = (r x v) / ||r x v||
      Shat = What x Rhat

    Returns
    -------
    C_eci_to_ric : (3,3) ndarray
        Rows are [Rhat; Shat; What].
    """
    r = np.asarray(r_eci, dtype=float).reshape(3,)
    v = np.asarray(v_eci, dtype=float).reshape(3,)

    rnorm = np.linalg.norm(r)
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)
    if rnorm < 1e-12 or hnorm < 1e-12:
        raise ValueError("Cannot build ECI->RIC rotation: degenerate state vector.")

    Rhat = r / rnorm
    What = h / hnorm
    Shat = np.cross(What, Rhat)

    return np.vstack((Rhat, Shat, What))
