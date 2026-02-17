import numpy as np


def state_noise_compensation(delta_t, n, m, Q):

    '''delta_t: time step (s)
    n: state dimension
    m: measurement dimension
    Q: process noise covariance (m x m)'''

    dt = float(delta_t)
    gamma = np.zeros((n, m), dtype=float)
    gamma[0:3, 0:3] = dt**2/2 * np.eye(3)
    gamma[3:6, 0:3] = dt * np.eye(3)

    snc_eci = gamma @ Q @ gamma.T
    # snc_ric = gamma.T @ Q @ gamma
    return snc_eci  #, snc_ric
