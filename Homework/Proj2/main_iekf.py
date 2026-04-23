import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append("../../")

from src.Functions.filters import ExtendedKalmanFilter
from src.Functions.stations import Stations
from src.Functions.srp_dyn_model import mu_sun_srp_state_deriv
from src.Functions.jacobians import srp_thirdbody_variational_eq
from src.Functions.Ephem import ephem
from src.helpers.plotting.post_processing import run_filter_post_processing
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.plot_state_estimate_3sigma import (
    make_all_state_3sigma_envelope_plot,
    make_cr_3sigma_plot,
)


def build_stations():
    theta0_deg = 0
    w_earth_rad_per_s = 7.29211585275553e-5
    radius_earth_km = 6378.1363

    return [
        Stations(
            "DSS 34",
            lat_deg=-35.398333,
            lon_deg=148.981944,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + 0.691750,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "DSS 65",
            lat_deg=40.427222,
            lon_deg=355.749444,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + 0.834539,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "DSS 13",
            lat_deg=35.247164,
            lon_deg=243.205000,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km + 1.07114904,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
    ]


# Same IC scaffold as UKF script: 7-state [r, v, Cr]
x0_iekf = np.array(
    [
        -274096770.76544,
        -92859266.4499061,
        -40199493.6677441,
        32.6704564599943,
        -8.93838913761049,
        -3.87881914050316,
        1.0,
    ],
    dtype=float,
)

P0_iekf = np.diag(
    [
        100.0**2,  # x [km]
        100.0**2,  # y [km]
        100.0**2,  # z [km]
        0.1**2,  # vx [km/s]
        0.1**2,  # vy [km/s]
        0.1**2,  # vz [km/s]
        0.1**2,  # Cr [-]
    ],
)


def build_problem_constants():
    Jd0 = 2456296.25
    mu_sun = 132712440017.987
    AU_km = 149597870.7
    solar_flux_W_m2 = 1357.0
    SRP_area_mass_ratio = 0.01

    pConst = type("pConst", (), {})()
    pConst.mu_earth = 398600.4415
    pConst.mu_sun = mu_sun

    scConst = type("scConst", (), {})()
    scConst.area = SRP_area_mass_ratio
    scConst.mass = 1.0
    scConst.solar_flux_1au = solar_flux_W_m2
    scConst.c = 299792458.0
    scConst.AU_m = AU_km * 1000.0

    def earth_state_func(tau):
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    def sun_state_func(tau):
        jd = Jd0 + float(tau) / 86400.0
        rE_km, vE_km_s, _ = ephem(jd, 3, frame="EME2000")
        rE_km = np.asarray(rE_km, dtype=float).reshape(3,)
        vE_km_s = np.asarray(vE_km_s, dtype=float).reshape(3,)
        return -rE_km, -vE_km_s

    return pConst, scConst, earth_state_func, sun_state_func


def load_project2_obs(obs_path: Path) -> list[dict]:
    """
    Read Project 2 observation file into filter format:
      {"station": str, "t": float, "rho_km": float, "rho_dot_km_s": float}
    """
    df = pd.read_csv(
        obs_path,
        skipinitialspace=True,
        index_col=False,
        usecols=range(7),
    )
    df.columns = [c.strip() for c in df.columns]

    col_time = "Time since Epoch"
    station_cols = {
        "DSS 34": ("DSS34 Range (km)", "DSS34 Range-Rate (km/sec)"),
        "DSS 65": ("DSS65 Range (km)", "DSS65 Range-Rate (km/sec)"),
        "DSS 13": ("DSS13 Range (km)", "DSS13 Range-Rate (km/sec)"),
    }

    pieces = []
    for st_name, (rho_col, rhod_col) in station_cols.items():
        part = df[[col_time, rho_col, rhod_col]].rename(
            columns={
                col_time: "t",
                rho_col: "rho_km",
                rhod_col: "rho_dot_km_s",
            }
        )
        part["station"] = st_name
        pieces.append(part)

    meas = pd.concat(pieces, ignore_index=True)
    meas["t"] = pd.to_numeric(meas["t"], errors="coerce")
    meas["rho_km"] = pd.to_numeric(meas["rho_km"], errors="coerce")
    meas["rho_dot_km_s"] = pd.to_numeric(meas["rho_dot_km_s"], errors="coerce")
    meas = meas.dropna(subset=["rho_km", "rho_dot_km_s"])
    meas = meas.sort_values("t")[["station", "t", "rho_km", "rho_dot_km_s"]]
    return meas.to_dict(orient="records")


def to_plotting_result_6state_from_filter_km(out: dict, *, R_km: np.ndarray):
    """
    Convert filter output in km/km-s units to shared plotting result format in m/m-s
    for 6-state plotting helpers (x,y,z,vx,vy,vz).
    """
    xhat_key = "xhat_meas" if "xhat_meas" in out else "Xhat_meas"
    phat_key = "P_meas" if "P_meas" in out else "Phat_meas"

    xhat = np.asarray(out[xhat_key], dtype=float)
    P = np.asarray(out[phat_key], dtype=float)

    xhat6_m = xhat[:, 0:6] * 1000.0
    P6_m = P[:, 0:6, 0:6] * (1000.0**2)
    two_sigma6_m = 2.0 * np.sqrt(np.maximum(np.diagonal(P6_m, axis1=1, axis2=2), 0.0))

    d = {
        "t_meas": np.asarray(out["t_meas"], dtype=float),
        "station_meas": list(out["station_meas"]),
        "xhat_meas": xhat6_m,
        "Xhat_meas": xhat6_m,
        "P_meas": P6_m,
        "two_sigma_meas": two_sigma6_m,
        "state_error_meas": None
        if out.get("state_error_meas", None) is None
        else np.asarray(out["state_error_meas"], dtype=float)[:, 0:6] * 1000.0,
        "prefit_resids_final": None
        if out.get("prefit_resids_final", None) is None
        else np.asarray(out["prefit_resids_final"], dtype=float) * 1000.0,
        "postfit_resids_linear_final": None
        if out.get("postfit_resids_linear_final", None) is None
        else np.asarray(out["postfit_resids_linear_final"], dtype=float) * 1000.0,
        "postfit_resids_meas": None
        if out.get("postfit_resids_meas", None) is None
        else np.asarray(out["postfit_resids_meas"], dtype=float) * 1000.0,
        "rms_final": out.get("rms_final", {}),
        "rms_by_iter": out.get("rms_by_iter", None),
        "R": np.asarray(R_km, dtype=float) * (1000.0**2),
    }
    return run_filter_post_processing(out=d)


def make_standard_plot_set(out: dict, outdir: Path, *, R_km: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    result = to_plotting_result_6state_from_filter_km(out, R_km=R_km)
    make_postfit_residuals_linear_plot(result, outdir)
    make_trace_cov_pos_vel_plot(result, outdir, length_unit="m")
    make_all_state_3sigma_envelope_plot(out, outdir)
    make_cr_3sigma_plot(out, outdir)


def select_meas_day_window(
    all_meas: list[dict],
    day_start: float,
    day_end: float,
    *,
    include_start: bool,
) -> list[dict]:
    t0 = float(day_start) * 86400.0
    t1 = float(day_end) * 86400.0
    eps = 1.0e-9

    sel = []
    for m in all_meas:
        tm = float(m["t"])
        if include_start:
            if (tm >= t0 - eps) and (tm <= t1 + eps):
                sel.append(m)
        else:
            if (tm > t0 + eps) and (tm <= t1 + eps):
                sel.append(m)
    return sel


def combine_chunk_outputs(chunk_outs: list[dict]) -> dict:
    if len(chunk_outs) == 0:
        raise ValueError("No chunk outputs provided.")

    out = dict(chunk_outs[-1])  # carry final metadata by default

    arr_keys = [
        "t_meas",
        "xhat_meas",
        "Xhat_meas",
        "X_pf",
        "P_meas",
        "P_pf",
        "two_sigma_meas",
        "prefit_resids_final",
        "postfit_resids_linear_final",
        "postfit_resids_meas",
        "X_pred_hist",
        "P_pred_hist",
        "Phi_step_hist",
    ]
    for k in arr_keys:
        arrs = [np.asarray(c[k], dtype=float) for c in chunk_outs if c.get(k, None) is not None]
        if len(arrs) == len(chunk_outs) and len(arrs) > 0:
            out[k] = np.concatenate(arrs, axis=0)

    if all(("station_meas" in c) and (c["station_meas"] is not None) for c in chunk_outs):
        out["station_meas"] = [s for c in chunk_outs for s in c["station_meas"]]

    if any(c.get("state_error_meas", None) is None for c in chunk_outs):
        out["state_error_meas"] = None
    else:
        out["state_error_meas"] = np.concatenate(
            [np.asarray(c["state_error_meas"], dtype=float) for c in chunk_outs],
            axis=0,
        )

    # Keep these from first/last chunk for consistency.
    out["R"] = np.asarray(chunk_outs[0].get("R", out.get("R")), dtype=float)
    out["rms_final"] = chunk_outs[-1].get("rms_final", out.get("rms_final"))
    out["rms_by_iter"] = chunk_outs[-1].get("rms_by_iter", out.get("rms_by_iter"))

    # Alias consistency
    if ("xhat_meas" not in out) and ("Xhat_meas" in out):
        out["xhat_meas"] = out["Xhat_meas"]
    if ("Xhat_meas" not in out) and ("xhat_meas" in out):
        out["Xhat_meas"] = out["xhat_meas"]

    return out


def main():
    obs_path = Path(__file__).resolve().parent / "Given_data" / "Project2b_Obs.txt"
    all_meas = load_project2_obs(obs_path)
    stations = build_stations()

    pConst, scConst, earth_state_func, sun_state_func = build_problem_constants()

    sigma_rho_km = 5.0e-3
    sigma_rhod_km_s = 0.5e-6
    R_iekf = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    # SNC
    sigma_acc_km_s2 = 1.0e-9
    Q_iekf = np.diag([sigma_acc_km_s2**2, sigma_acc_km_s2**2, sigma_acc_km_s2**2])

    dyn_iekf = lambda tau, x: mu_sun_srp_state_deriv(
        t=tau,
        X=x,
        pConst=pConst,
        scConst=scConst,
        earth_state_func=earth_state_func,
        sun_state_func=sun_state_func,
    )

    def jac_iekf(tau, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        r_earth, _ = earth_state_func(tau)
        r_sun, _ = sun_state_func(tau)
        return srp_thirdbody_variational_eq(
            r_sc=x[0:3],
            r_earth=r_earth,
            r_sun=r_sun,
            Cr=float(x[6]),
            area=scConst.area,
            mass=scConst.mass,
            mu_earth=pConst.mu_earth,
            mu_i=pConst.mu_sun,
            solar_flux_1au=scConst.solar_flux_1au,
            c=scConst.c,
            AU_m=scConst.AU_m,
        )

    iekf = ExtendedKalmanFilter(
        x0=x0_iekf,
        P0=P0_iekf,
        R=R_iekf,
        Q=Q_iekf,
        mu=pConst.mu_earth,
        J2=0.0,
        J3=0.0,
        Re=6378.1363,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="RK45",
        j2=False,
        j3=False,
        first_pass_gap_s=6 * 3600.0,
        dyn_fun=dyn_iekf,
        dyn_jac=jac_iekf,
    )

    # Iteration settings for IEKF chunks.
    max_iter = 10
    iter_tol = 1.0e-8

    plot_root = Path(__file__).resolve().parent / "Plots" / "IEKF"
    chunk_root = plot_root / "Chunks"
    total_root = plot_root / "Total Run"
    chunk_root.mkdir(parents=True, exist_ok=True)
    total_root.mkdir(parents=True, exist_ok=True)

    all_meas = sorted(all_meas, key=lambda m: float(m["t"]))
    all_days = np.array([float(m["t"]) for m in all_meas], dtype=float) / 86400.0
    max_day = float(np.max(all_days))

    chunk_outs = []
    day_start = 0.0
    chunk_i = 1
    t_prev_init = None

    while day_start < (max_day - 1.0e-12):
        day_end = min(day_start + 50.0, max_day)
        meas_chunk = select_meas_day_window(
            all_meas,
            day_start,
            day_end,
            include_start=(chunk_i == 1),
        )

        if len(meas_chunk) == 0:
            day_start = day_end
            chunk_i += 1
            continue

        print(
            f"\nRunning IEKF Chunk {chunk_i:02d}: Days {day_start:.3f} To {day_end:.3f} "
            f"({len(meas_chunk)} measurements)"
        )

        out_chunk = iekf.run(
            all_meas=meas_chunk,
            stations=stations,
            Xtrue_meas=None,
            t_prev_init=t_prev_init,
            show_progress=True,
            progress_every=50,
            iterated=True,
            max_iter=max_iter,
            iter_tol=iter_tol,
        )

        chunk_outs.append(out_chunk)
        t_prev_init = float(out_chunk["t_meas"][-1])

        d0 = int(np.floor(day_start + 1.0e-9))
        d1 = int(np.ceil(day_end - 1.0e-9))
        chunk_dir = chunk_root / f"Days_{d0:03d}_{d1:03d}"
        make_standard_plot_set(out_chunk, chunk_dir, R_km=R_iekf)
        print(
            f"Saved Chunk {chunk_i:02d} Plots: Days {day_start:.3f} To {day_end:.3f} "
            f"({len(out_chunk['t_meas'])} measurements)"
        )

        day_start = day_end
        chunk_i += 1

    out_total = combine_chunk_outputs(chunk_outs)
    print(f"\nIEKF Chunked Run Complete. Number Of Updates: {len(out_total['t_meas'])}")

    make_standard_plot_set(out_total, total_root, R_km=R_iekf)

    xf = np.asarray(out_total["xhat_meas"][-1], dtype=float).reshape(-1)
    print("\nFinal State Estimate (IEKF, Chunked):")
    print(f"  Position [km]    : [{xf[0]:.6f}, {xf[1]:.6f}, {xf[2]:.6f}]")
    print(f"  Velocity [km/s]  : [{xf[3]:.9f}, {xf[4]:.9f}, {xf[5]:.9f}]")
    print(f"  Cr [-]           : {xf[6]:.9f}")
    print(f"\nSaved IEKF Plots To: {plot_root}")


if __name__ == "__main__":
    main()

