import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def ellipse_points_from_cov(cov2: np.ndarray, n_sigma: float = 3.0, n_pts: int = 500):
    cov = np.asarray(cov2, dtype=float).reshape(2, 2)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    radii = float(n_sigma) * np.sqrt(vals)
    th = np.linspace(0.0, 2.0 * np.pi, int(n_pts))
    circ = np.vstack((np.cos(th), np.sin(th)))
    xy = vecs @ np.diag(radii) @ circ
    return xy[0, :], xy[1, :]
def extract_case_from_json(seed_block: dict, label: str):
    x_key = "final_state_ilkf_forward"
    p_key = "final_cov_ilkf_forward"
    t_key = "final_time_s_ilkf_forward"

    if x_key not in seed_block or p_key not in seed_block or t_key not in seed_block:
        raise ValueError(
            f"JSON is missing '{x_key}', '{p_key}', or '{t_key}' for case '{label}'. "
            "Re-run sixstate_main_end_ilkf.py to regenerate final_states_maneuver_seed_comparison.json."
        )

    x6 = np.asarray(seed_block[x_key], dtype=float).reshape(-1)
    p6 = np.asarray(seed_block[p_key], dtype=float)
    t0 = float(seed_block[t_key])

    if x6.shape != (6,):
        raise ValueError(f"Case '{label}' has invalid state shape {x6.shape}, expected (6,).")
    if p6.shape != (6, 6):
        raise ValueError(f"Case '{label}' has invalid covariance shape {p6.shape}, expected (6, 6).")
    if not np.all(np.isfinite(x6)) or not np.all(np.isfinite(p6)) or (not np.isfinite(t0)):
        raise ValueError(f"Case '{label}' has non-finite state/covariance/time.")

    return x6, p6, t0


def augment_to_7state(x6: np.ndarray, p6: np.ndarray):
    x7 = np.zeros(7, dtype=float)
    x7[:6] = np.asarray(x6, dtype=float).reshape(6,)
    x7[6] = CR_FIXED

    p7 = np.zeros((7, 7), dtype=float)
    p7[:6, :6] = np.asarray(p6, dtype=float).reshape(6, 6)
    p7[6, 6] = 1.0e-12
    return x7, p7


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
    Phi_start_to_3soi = y_3soi[7:].reshape(7, 7)
    P_3soi = Phi_start_to_3soi @ np.asarray(P7, dtype=float) @ Phi_start_to_3soi.T
    return t_3soi, x_3soi, P_3soi


def _safe_label_for_filename(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(label)).strip("_")


def make_bplane_plot(rec: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("T [km]")
    ax.set_ylabel("R [km]")
    ax.set_title("Part 3: B-Plane Crossings and 3-Sigma Ellipses (2b Estimates)")

    cov_tr = np.asarray(rec["cov_tr_km2"], dtype=float)
    ex, ey = ellipse_points_from_cov(cov_tr, n_sigma=3.0, n_pts=500)

    bt = float(rec["BdotT_km"])
    br = float(rec["BdotR_km"])
    label = str(rec["label"])

    ax.plot(bt + ex, br + ey, "-", color="tab:blue", linewidth=1.5, label=f"3-sigma: {label}")
    ax.plot(bt, br, "x", color="tab:blue", markersize=9, linewidth=2.0, label=f"{label} crossing")

    ax.legend(loc="best")
    fig.tight_layout()
    file_tag = _safe_label_for_filename(label)
    fig.savefig(outdir / f"part3_bplane_crossings_3sigma_{file_tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    json_path = base / "Plots" / "End ILKF Sixstate" / "final_states_maneuver_seed_comparison.json"
    outdir = base / "Plots" / "Part3 BPlane"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    p_const, sc_const, earth_state_func, sun_state_func = build_problem_constants()

    cases = []
    sf = payload.get("seed_forward_iekf", {})
    ss = payload.get("seed_smoothed_iekf", {})

    # Keep only the two requested pipelines:
    # 1) seeded from forward IEKF history
    # 2) seeded from smoothed IEKF history
    # ILKF forward/smoothed finals are effectively identical in your runs, so we
    # intentionally use only final_state_ilkf_forward for each seed path.
    if len(sf) > 0:
        x6_fwd, p6_fwd, t0_fwd = extract_case_from_json(sf, "Forward IEKF Seed")
        cases.append(("Forward IEKF Seed", x6_fwd, p6_fwd, t0_fwd))
    if len(ss) > 0:
        x6_s, p6_s, t0_s = extract_case_from_json(ss, "Smoothed IEKF Seed")
        cases.append(("Smoothed IEKF Seed", x6_s, p6_s, t0_s))

    if len(cases) == 0:
        raise ValueError("No ILKF final states found in JSON.")

    records = []
    for label, x6, p6, t0 in cases:
        x7, p7 = augment_to_7state(x6, p6)
        t_3soi, x_3soi, p_3soi = propagate_snapshot_to_3soi_with_stm(
            x7=x7,
            P7=p7,
            t_start=float(t0),
            p_const=p_const,
            sc_const=sc_const,
            earth_state_func=earth_state_func,
            sun_state_func=sun_state_func,
            t_search_days=400.0,
        )

        (
            bdot_r_km,
            bdot_t_km,
            _sig_r,
            _sig_t,
            _sig_rt,
            _x_crossing,
            p_bplane,
            _str2eci,
            _xphi_bplane,
            _t_bplane,
        ) = calc_bplane(
            XPhi_3SOI=x_3soi,
            t_3SOI=t_3soi,
            P_3SOI=p_3soi,
            pConst=p_const,
            scConst=sc_const,
            earth_state_func=earth_state_func,
            sun_state_func=sun_state_func,
        )

        cov_tr = np.array(
            [
                [p_bplane[1, 1], p_bplane[1, 2]],
                [p_bplane[1, 2], p_bplane[2, 2]],
            ],
            dtype=float,
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

        records.append(
            {
                "label": label,
                "BdotT_km": float(bdot_t_km),
                "BdotR_km": float(bdot_r_km),
                "rp_km": rp_km,
                "rp_alt_km": rp_alt_km,
                "sigma_T_km": float(np.sqrt(max(cov_tr[0, 0], 0.0))),
                "sigma_R_km": float(np.sqrt(max(cov_tr[1, 1], 0.0))),
                "cov_tr_km2": cov_tr,
            }
        )

        print(
            f"{label}: "
            f"BdotT={float(bdot_t_km):10.3f} km, "
            f"BdotR={float(bdot_r_km):10.3f} km, "
            f"rp={records[-1]['rp_km']:10.3f} km, "
            f"alt={records[-1]['rp_alt_km']:10.3f} km"
        )

    for rec in records:
        make_bplane_plot(rec, outdir)

    rows = []
    for rec in records:
        rows.append(
            {
                "label": rec["label"],
                "BdotT_km": rec["BdotT_km"],
                "BdotR_km": rec["BdotR_km"],
                "rp_km": rec["rp_km"],
                "rp_alt_km": rec["rp_alt_km"],
                "sigma_T_km": rec["sigma_T_km"],
                "sigma_R_km": rec["sigma_R_km"],
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "part3_bplane_summary.csv", index=False)
    print(f"Saved B-plane plots to: {outdir}")
    print(f"Saved B-plane summary: {outdir / 'part3_bplane_summary.csv'}")


if __name__ == "__main__":
    main()
