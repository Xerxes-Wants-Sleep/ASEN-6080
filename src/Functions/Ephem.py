import numpy as np
from dataclasses import dataclass


DEG2RAD = np.pi / 180.0
AU_KM = 149_597_870.700
MU_SUN_KM = 132_712_440_017.987
OBLIQUITY_EME2000_RAD = 23.4393 * DEG2RAD


@dataclass
class EphemCoeff:
    L: np.ndarray
    a: np.ndarray
    e: np.ndarray
    i: np.ndarray
    W: np.ndarray
    P: np.ndarray
    mu_p: float


def coes_to_rv(
    a: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    nu: float,
    mu: float,
):
    """
    Convert classical orbital elements to Cartesian state.

    Units:
        a in km
        angles in rad
        mu in km^3/s^2

    Returns:
        R, V as (3,) ndarrays in km and km/s
    """
    p = a * (1.0 - e**2)

    r_pf = np.array([
        p * np.cos(nu) / (1.0 + e * np.cos(nu)),
        p * np.sin(nu) / (1.0 + e * np.cos(nu)),
        0.0,
    ])

    v_pf = np.array([
        -np.sqrt(mu / p) * np.sin(nu),
        np.sqrt(mu / p) * (e + np.cos(nu)),
        0.0,
    ])

    cO = np.cos(raan)
    sO = np.sin(raan)
    ci = np.cos(inc)
    si = np.sin(inc)
    cw = np.cos(argp)
    sw = np.sin(argp)

    # Perifocal -> inertial
    C = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])

    R = C @ r_pf
    V = C @ v_pf
    return R, V


def ephemeride_coeff(planet: int) -> EphemCoeff:
    """
    Planet numbering matches the MATLAB:
        1 Mercury
        2 Venus
        3 Earth
        4 Mars
        5 Jupiter
        6 Saturn
        7 Uranus
        8 Neptune
        9 Pluto
    """
    if planet == 1:
        L = np.array([252.250906, 149472.6746358, -0.00000535, 0.000000002])
        a = np.array([0.387098310, 0.0, 0.0, 0.0])
        e = np.array([0.20563175, 0.000020406, -0.0000000284, -0.00000000017])
        i = np.array([7.004986, -0.0059516, 0.00000081, 0.000000041])
        W = np.array([48.330893, -0.1254229, -0.00008833, -0.000000196])
        P = np.array([77.456119, 0.1588643, -0.00001343, 0.000000039])
        mu_p = 2.20320804864179e4
    elif planet == 2:
        L = np.array([181.979801, 58517.8156760, 0.00000165, -0.000000002])
        a = np.array([0.72332982, 0.0, 0.0, 0.0])
        e = np.array([0.00677188, -0.000047766, 0.0000000975, 0.00000000044])
        i = np.array([3.394662, -0.0008568, -0.00003244, 0.000000010])
        W = np.array([76.679920, -0.2780080, -0.00014256, -0.000000198])
        P = np.array([131.563707, 0.0048646, -0.00138232, -0.000005332])
        mu_p = 3.2485859882646e5
    elif planet == 3:
        L = np.array([100.466449, 35999.3728519, -0.00000568, 0.0])
        a = np.array([1.000001018, 0.0, 0.0, 0.0])
        e = np.array([0.01670862, -0.000042037, -0.0000001236, 0.00000000004])
        i = np.array([0.0, 0.0130546, -0.00000931, -0.000000034])
        W = np.array([174.873174, -0.2410908, 0.00004067, -0.000001327])
        P = np.array([102.937348, 0.3225557, 0.00015026, 0.000000478])
        mu_p = 3.98600432896939e5
    elif planet == 4:
        L = np.array([355.433275, 19140.2993313, 0.00000261, -0.000000003])
        a = np.array([1.523679342, 0.0, 0.0, 0.0])
        e = np.array([0.09340062, 0.000090483, -0.0000000806, -0.00000000035])
        i = np.array([1.849726, -0.0081479, -0.00002255, -0.000000027])
        W = np.array([49.558093, -0.2949846, -0.00063993, -0.000002143])
        P = np.array([336.060234, 0.4438898, -0.00017321, 0.000000300])
        mu_p = 4.28283142580671e4
    elif planet == 5:
        L = np.array([34.351484, 3034.9056746, -0.00008501, 0.000000004])
        a = np.array([5.202603191, 0.0000001913, 0.0, 0.0])
        e = np.array([0.04849485, 0.000163244, -0.0000004719, -0.00000000197])
        i = np.array([1.303270, -0.0019872, 0.00003318, 0.000000092])
        W = np.array([100.464441, 0.1766828, 0.00090387, -0.000007032])
        P = np.array([14.331309, 0.2155525, 0.00072252, -0.000004590])
        mu_p = 1.26712767857796e8
    elif planet == 6:
        L = np.array([50.077471, 1222.1137943, 0.00021004, -0.000000019])
        a = np.array([9.554909596, -0.0000021389, 0.0, 0.0])
        e = np.array([0.05550862, -0.000346818, -0.0000006456, 0.00000000338])
        i = np.array([2.488878, 0.0025515, -0.00004903, 0.000000018])
        W = np.array([113.665524, -0.2566649, -0.00018345, 0.000000357])
        P = np.array([93.056787, 0.5665496, 0.00052809, 0.000004882])
        mu_p = 3.79406260611373e7
    elif planet == 7:
        L = np.array([314.055005, 428.4669983, -0.00000486, 0.000000006])
        a = np.array([19.218446062, -0.0000000372, 0.00000000098, 0.0])
        e = np.array([0.04629590, -0.000027337, 0.0000000790, 0.00000000025])
        i = np.array([0.773196, -0.0016869, 0.00000349, 0.000000016])
        W = np.array([74.005947, 0.0741461, 0.00040540, 0.000000104])
        P = np.array([173.005159, 0.0893206, -0.00009470, 0.000000413])
        mu_p = 5.79454900707188e6
    elif planet == 8:
        L = np.array([304.348665, 218.4862002, 0.00000059, -0.000000002])
        a = np.array([30.110386869, -0.0000001663, 0.00000000069, 0.0])
        e = np.array([0.00898809, 0.000006408, -0.0000000008, -0.00000000005])
        i = np.array([1.769952, 0.0002257, 0.00000023, 0.0])
        W = np.array([131.784057, -0.0061651, -0.00000219, -0.000000078])
        P = np.array([48.123691, 0.0291587, 0.00007051, -0.000000023])
        mu_p = 6.83653406387926e6
    elif planet == 9:
        L = np.array([238.92903833, 145.20780515, 0.0, 0.0])
        a = np.array([39.48211675, -0.00031596, 0.0, 0.0])
        e = np.array([0.24882730, 0.00005170, 0.0, 0.0])
        i = np.array([17.14001206, 0.00004818, 0.0, 0.0])
        W = np.array([110.30393684, -0.01183482, 0.0, 0.0])
        P = np.array([224.06891629, -0.04062942, 0.0, 0.0])
        mu_p = 9.81600887707005e2
    else:
        raise ValueError("Planet must be an integer from 1 to 9.")

    return EphemCoeff(L=L, a=a, e=e, i=i, W=W, P=P, mu_p=mu_p)


def ephem(jd: float, planet: int, frame: str = "EME2000"):
    """
    Python version of the provided MATLAB Ephem().

    Inputs
    ------
    jd : float
        Julian Day
    planet : int
        1..9
    frame : str
        'EME2000' or 'EMO2000'

    Returns
    -------
    R : (3,) ndarray
        Planet position in km
    V : (3,) ndarray
        Planet velocity in km/s
    mu_p : float
        Planet gravitational parameter in km^3/s^2
    """
    T = (jd - 2451545.0) / 36525.0
    coeff = ephemeride_coeff(planet)

    Tvec = np.array([1.0, T, T * T, T * T * T])

    L = np.dot(coeff.L, Tvec) * DEG2RAD
    a = np.dot(coeff.a, Tvec) * AU_KM
    e = np.dot(coeff.e, Tvec)
    inc = np.dot(coeff.i, Tvec) * DEG2RAD
    W = np.dot(coeff.W, Tvec) * DEG2RAD
    P = np.dot(coeff.P, Tvec) * DEG2RAD

    w = P - W
    M = L - P

    Ccen = (
        (2 * e - e**3 / 4 + 5 * e**5 / 96) * np.sin(M)
        + (5 * e**2 / 4 - 11 * e**4 / 24) * np.sin(2 * M)
        + (13 * e**3 / 12 - 43 * e**5 / 64) * np.sin(3 * M)
        + (103 * e**4 / 96) * np.sin(4 * M)
        + (1097 * e**5 / 960) * np.sin(5 * M)
    )

    nu = M + Ccen

    R, V = coes_to_rv(a, e, inc, W, w, nu, MU_SUN_KM)

    frame_upper = frame.upper()
    if frame_upper == "EME2000":
        theta = OBLIQUITY_EME2000_RAD
        C = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ])
        R = C @ R
        V = C @ V
    elif frame_upper != "EMO2000":
        raise ValueError("frame must be 'EME2000' or 'EMO2000'.")

    return R, V, coeff.mu_p