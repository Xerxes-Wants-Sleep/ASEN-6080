import numpy as np

def Keplarian_to_Cartesian(mu, a, e, i_deg, RAAN_deg, w_deg, f_deg):
    """
    Convert orbital elements to inertial Cartesian state vectors
    using direct formulas from the slides.
    
    Inputs:
        mu      - gravitational parameter [km^3/s^2]
        a       - semi-major axis [km]
        e       - eccentricity
        i_deg   - inclination [deg]
        RAAN_deg - Right Ascension of Ascending Node [deg]
        w_deg   - argument of periapsis [deg]
        f_deg   - true anomaly [deg]
    
    Returns:
        r_N - inertial position vector [km]
        v_N - inertial velocity vector [km/s]
        NP  - rotation matrix from inertial to perifocal
    """
    # Convert to radians
    i = np.radians(i_deg)
    RAAN = np.radians(RAAN_deg)
    w = np.radians(w_deg)
    f = np.radians(f_deg)

    # Unperturbed Orbit Constants
    p = a * (1 - e**2) # Semi-Latus Rectum
    r_mag = p / (1 + e*np.cos(f)) # Orbit Radius
    h = np.sqrt(mu * p)  # Angular Momentum

    # True Longitude
    theta = w + f

    # Inertial Position
    r_N = r_mag * np.array([
        np.cos(RAAN)*np.cos(theta) - np.sin(RAAN)*np.sin(theta)*np.cos(i),
        np.sin(RAAN)*np.cos(theta) + np.cos(RAAN)*np.sin(theta)*np.cos(i),
        np.sin(theta)*np.sin(i)
    ])

    # Inertial Vel
    v_N = (mu/h) * np.array([
        -np.cos(RAAN)*(np.sin(theta) + e*np.sin(w)) - np.sin(RAAN)*(np.cos(theta) + e*np.cos(w))*np.cos(i),
        -np.sin(RAAN)*(np.sin(theta) + e*np.sin(w)) + np.cos(RAAN)*(np.cos(theta) + e*np.cos(w))*np.cos(i),
         (np.cos(theta) + e*np.cos(w))*np.sin(i)
    ])

    # PN DCM
    PN = np.array([
        [np.cos(w)*np.cos(RAAN) - np.sin(w)*np.cos(i)*np.sin(RAAN),
         np.cos(w)*np.sin(RAAN) + np.sin(w)*np.cos(i)*np.cos(RAAN),
         np.sin(w)*np.sin(i)],
        [-np.sin(w)*np.cos(RAAN) - np.cos(w)*np.cos(i)*np.sin(RAAN),
         -np.sin(w)*np.sin(RAAN) + np.cos(w)*np.cos(i)*np.cos(RAAN),
         np.cos(w)*np.sin(i)],
        [np.sin(i)*np.sin(RAAN),
         -np.sin(i)*np.cos(RAAN),
         np.cos(i)]
    ])

    return r_N, v_N, PN
