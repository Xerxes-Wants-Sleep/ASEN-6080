import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.Functions.Ephem import ephem
from src.Functions.SOIcheck import SOIcheck
from src.Functions.calcB_plane import calc_bplane
from src.Functions.srp_dyn_model import mu_sun_srp_stm_deriv


CR_FIXED = 1.38
R_EARTH_KM = 6378.1363

# Image-provided 6-state [km, km/s]
X0_6 = np.array(
    [
        -24626021.4210292,
        -14913429.1617499,
        -6491135.4071616,
        10.5441887513048,
        4.46497188342413,
        1.94198580346455,
    ],
    dtype=float,
)

# Image-provided 6x6 covariance (km-based units)
P0_6 = np.array(
    [
        [0.0147750790235536, -0.0163408609582470, -0.0185902272282621, -3.91997705500907e-09, 1.18838319808702e-08, -6.01526449860888e-09],
        [-0.0163408609601400, 0.5807647244753250, -1.2706736667798700, 1.39231113716277e-08, 2.05672984290544e-07, -5.27790589985974e-07],
        [-0.0185902272239188, -1.2706736667870300, 2.9864441355649600, -1.70445035024806e-08, -5.17110528884925e-07, 1.23396813014130e-06],
        [-3.91997705496925e-09, 1.39231113710818e-08, -1.70445035013795e-08, 1.49404719866563e-15, 4.07657561167058e-16, -8.10951256081816e-15],
        [1.18838319849014e-08, 2.05672984291314e-07, -5.17110528902009e-07, 4.07657560009307e-16, 1.16349562873578e-13, -2.61998430811561e-13],
        [-6.01526450801953e-09, -5.27790589984852e-07, 1.23396813017448e-06, -8.10951255817589e-15, -2.61998430817686e-13, 6.25528204386538e-13],
    ],
    dtype=float,
)


def build_problem_constants():
    jd0 = 2456296.25
    mu_sun = 132712440017.987
    au_km = 149597870.7
    solar_flux_w_m2 = 1357.0
    srp_area_mass_ratio = 0.01

    p_const = type("pConst", (), {})()
    p_const.mu_earth = 3.98600432896939e5
    p_const.mu_sun = mu_sun

    sc_const = type("scConst", (), {})()
    sc_const.area = srp_area_mass_ratio
    sc_const.mass = 1.0
    sc_const.solar_flux_1au = solar_flux_w_m2
    sc_const.c = 299792458.0
    sc_const.AU_m = au_km * 1000.0

    def earth_state_func(tau):
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    def sun_state_func(tau):
        jd = jd0 + float(tau) / 86400.0
        r_e_km, v_e_km_s, _ = ephem(jd, 3, frame="EME2000")
        r_e_km = np.asarray(r_e_km, dtype=float).reshape(3,)
        v_e_km_s = np.asarray(v_e_km_s, dtype=float).reshape(3,)
        return -r_e_km, -v_e_km_s

    return p_const, sc_const, earth_state_func, sun_state_func


def augment_to_7state(x6: np.ndarray, p6: np.ndarray):
    x7 = np.zeros(7, dtype=float)
    x7[:6] = np.asarray(x6, dtype=float).reshape(6,)
    x7[6] = CR_FIXED

    p7 = np.zeros((7, 7), dtype=float)
    p7[:6, :6] = np.asarray(p6, dtype=float).reshape(6, 6)
    p7[6, 6] = 1.0e-12
    return x7, p7


def propagate_snapshot_to_3soi_with_stm(
    *,
    x7: np.ndarray,
    P7: np.ndarray,
    t_start: float,
    p_const,
    sc_const,
    earth_state_func,
    sun_state_func,
    t_search_days: float = 400.0,
):
    y0 = np.hstack((np.asarray(x7, dtype=float).reshape(7), np.eye(7).reshape(-1)))

    soi_event = lambda tau, y: SOIcheck(tau, y[:7])
    soi_event.terminal = True
    soi_event.direction = -1.0

    sol = solve_ivp(
        fun=lambda tau, y: mu_sun_srp_stm_deriv(
            t=tau,
            XPhi=y,
            pConst=p_const,
            scConst=sc_const,
            earth_state_func=earth_state_func,
            sun_state_func=sun_state_func,
        ),
        t_span=(float(t_start), float(t_start) + float(t_search_days) * 86400.0),
        y0=y0,
        events=soi_event,
        rtol=1.0e-10,
        atol=1.0e-10,
        method="RK45",
        max_step=3600.0,
    )

    if (not sol.success) or (len(sol.t_events[0]) == 0):
        raise RuntimeError(f"3*RSOI crossing not found from t={float(t_start):.3f} s.")

    t_3soi = float(sol.t_events[0][0])
    y_3soi = np.asarray(sol.y_events[0][0], dtype=float)
    x_3soi = y_3soi[:7]
    phi_start_to_3soi = y_3soi[7:].reshape(7, 7)
    P_3soi = phi_start_to_3soi @ np.asarray(P7, dtype=float) @ phi_start_to_3soi.T
    return t_3soi, x_3soi, P_3soi


def compute_vinf_km_s_from_state(*, x_3soi: np.ndarray, t_3soi: float, mu_earth: float, earth_state_func) -> float:
    r_earth, v_earth = earth_state_func(float(t_3soi))
    r_rel = np.asarray(x_3soi[:3], dtype=float) - np.asarray(r_earth, dtype=float)
    v_rel = np.asarray(x_3soi[3:6], dtype=float) - np.asarray(v_earth, dtype=float)

    r_mag = float(np.linalg.norm(r_rel))
    v_mag2 = float(np.dot(v_rel, v_rel))
    v_inf2 = v_mag2 - 2.0 * float(mu_earth) / r_mag
    if v_inf2 <= 0.0:
        raise ValueError(f"Encounter is not hyperbolic (v_inf^2={v_inf2:.6e}).")
    return float(np.sqrt(v_inf2))


def compute_periapsis_km_from_bplane(
    *,
    BdotT_km: float,
    BdotR_km: float,
    v_inf_km_s: float,
    mu_earth: float,
) -> tuple[float, float]:
    """
    Compute periapsis radius and altitude from B-plane parameters.
    """
    B = float(np.hypot(float(BdotT_km), float(BdotR_km)))
    v2 = float(v_inf_km_s) ** 2
    ratio = B * v2 / float(mu_earth)
    e = float(np.sqrt(1.0 + ratio**2))
    rp = (float(mu_earth) / v2) * (e - 1.0)
    alt = rp - R_EARTH_KM
    return float(rp), float(alt)


def main():
    parser = argparse.ArgumentParser(description="B-plane sanity test using hardcoded image state/covariance.")
    parser.add_argument(
        "--t0-s",
        type=float,
        default=21376768.0,
        help="Cutoff time [s since epoch] for the hardcoded state/covariance.",
    )
    args = parser.parse_args()

    p_const, sc_const, earth_state_func, sun_state_func = build_problem_constants()

    x7, P7 = augment_to_7state(X0_6, P0_6)
    t_3soi, x_3soi, P_3soi = propagate_snapshot_to_3soi_with_stm(
        x7=x7,
        P7=P7,
        t_start=float(args.t0_s),
        p_const=p_const,
        sc_const=sc_const,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
        t_search_days=400.0,
    )

    (
        bdot_r_km,
        bdot_t_km,
        sig_r_km,
        sig_t_km,
        _sig_rt,
        _x_crossing,
        _p_bplane,
        _str2eci,
        _xphi_bplane,
        _t_bplane,
    ) = calc_bplane(
        XPhi_3SOI=x_3soi,
        t_3SOI=t_3soi,
        P_3SOI=P_3soi,
        pConst=p_const,
        scConst=sc_const,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    v_inf_km_s = compute_vinf_km_s_from_state(
        x_3soi=x_3soi,
        t_3soi=t_3soi,
        mu_earth=p_const.mu_earth,
        earth_state_func=earth_state_func,
    )
    rp_km, rp_alt_km = compute_periapsis_km_from_bplane(
        BdotT_km=float(bdot_t_km),
        BdotR_km=float(bdot_r_km),
        v_inf_km_s=v_inf_km_s,
        mu_earth=p_const.mu_earth,
    )

    print(
        "Image Test: "
        f"BdotT={float(bdot_t_km):10.3f} km, "
        f"BdotR={float(bdot_r_km):10.3f} km, "
        f"rp={rp_km:10.3f} km, "
        f"alt={rp_alt_km:10.3f} km"
    )
    print(
        "Image Test 1-sigma: "
        f"sigma_T={float(sig_t_km):10.3f} km, "
        f"sigma_R={float(sig_r_km):10.3f} km"
    )


if __name__ == "__main__":
    main()
