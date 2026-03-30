import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../../")

from src.Functions.consider_cov_filter import sequential_consider_cov_analysis
from src.Functions.stations import Stations
from src.helpers.plotting.common import as_hours, savefig
from src.helpers.plotting.plot_postfit_residuals_linear import make_postfit_residuals_linear_plot
from src.helpers.plotting.plot_trace_cov_pos_vel import make_trace_cov_pos_vel_plot
from src.helpers.plotting.post_processing import run_filter_post_processing


def build_stations():
    theta0_deg = 122.0
    w_earth_rad_per_s = 2.0 * np.pi / (3600.0 * 24.0)
    radius_earth_km = 6378.0

    return [
        Stations(
            "Station 1",
            lat_deg=-35.398333,
            lon_deg=148.981944,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "Station 2",
            lat_deg=40.427222,
            lon_deg=355.749444,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
        Stations(
            "Station 3",
            lat_deg=35.247164,
            lon_deg=243.205000,
            theta0_deg=theta0_deg,
            radius_earth=radius_earth_km,
            w_earth_rad_per_s=w_earth_rad_per_s,
        ),
    ]


def load_hw2_setup(num_meas: int | None = None):
    script_dir = Path(__file__).resolve().parent

    meas_path = script_dir.parent / "Homework2" / "meas_data" / "prob2_hw2_measurements_noisy.csv"
    truth_path = script_dir.parent / "Homework1" / "HW2_j3_on_truth.csv"

    df_meas = pd.read_csv(meas_path)
    all_meas = sorted(df_meas.to_dict(orient="records"), key=lambda m: float(m["t"]))
    if num_meas is not None:
        if num_meas < 2:
            raise ValueError("num_meas must be >= 2")
        all_meas = all_meas[:num_meas]

    truth_df = pd.read_csv(truth_path)
    truth_times = truth_df["t_s"].to_numpy(float)
    truth_states = truth_df[["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
    x0_true = truth_states[0, :]

    dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)*0
    X0_star = x0_true + dx

    sigma_rho_km = 1.0e-3
    sigma_rhod_km_s = 1.0e-6
    R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    stations = build_stations()

    return all_meas, truth_times, truth_states, stations, R, X0_star


def interp_truth_at_times(t_s: np.ndarray, truth_times: np.ndarray, truth_states: np.ndarray) -> np.ndarray:
    t_s = np.asarray(t_s, dtype=float)
    return np.column_stack([np.interp(t_s, truth_times, truth_states[:, k]) for k in range(6)])


def convert_cc_outputs_to_plot_units(
    out_cc: dict,
    *,
    truth_times: np.ndarray,
    truth_states_km: np.ndarray,
    station_meas: list[str],
    R_km: np.ndarray,
):
    t_meas = np.asarray(out_cc["t_meas"], dtype=float)

    XEst_km = np.asarray(out_cc["XEst"], dtype=float)
    XcEst_km = np.asarray(out_cc["XcEst"], dtype=float)
    P_km = np.asarray(out_cc["PEst"], dtype=float)
    Pc_km = np.asarray(out_cc["PcEst"], dtype=float)

    prefit_km = np.asarray(out_cc["prefit_res"], dtype=float)
    postfit_km = np.asarray(out_cc["postfit_res"], dtype=float)

    Xtrue_km = interp_truth_at_times(t_meas, truth_times, truth_states_km)

    XEst_m = XEst_km * 1000.0
    XcEst_m = XcEst_km * 1000.0
    Xtrue_m = Xtrue_km * 1000.0

    P_m = P_km * (1000.0**2)
    Pc_m = Pc_km * (1000.0**2)

    state_error_m = XEst_m - Xtrue_m
    state_error_c_m = XcEst_m - Xtrue_m

    prefit_m = prefit_km * 1000.0
    postfit_m = postfit_km * 1000.0
    R_m = np.asarray(R_km, dtype=float) * (1000.0**2)

    base_plot_out = {
        "t_meas": t_meas,
        "station_meas": station_meas,
        "xhat_meas": XEst_m,
        "P_meas": P_m,
        "two_sigma_meas": 2.0 * np.sqrt(np.maximum(np.diagonal(P_m, axis1=1, axis2=2), 0.0)),
        "state_error_meas": state_error_m,
        "prefit_resids_final": prefit_m,
        "postfit_resids_linear_final": postfit_m,
        "postfit_resids_meas": postfit_m,
        "rms_final": {},
        "R": R_m,
    }

    p_plot = run_filter_post_processing(out=base_plot_out)

    pc_plot_out = dict(base_plot_out)
    pc_plot_out["P_meas"] = Pc_m
    pc_plot_out["two_sigma_meas"] = 2.0 * np.sqrt(np.maximum(np.diagonal(Pc_m, axis1=1, axis2=2), 0.0))
    pc_plot = run_filter_post_processing(out=pc_plot_out)

    return {
        "t_meas": t_meas,
        "XEst_m": XEst_m,
        "XcEst_m": XcEst_m,
        "Xtrue_m": Xtrue_m,
        "state_error_m": state_error_m,
        "state_error_c_m": state_error_c_m,
        "P_m": P_m,
        "Pc_m": Pc_m,
        "prefit_m": prefit_m,
        "postfit_m": postfit_m,
        "p_plot": p_plot,
        "pc_plot": pc_plot,
    }


def _plot_state_error_with_two_sigma(
    t_s: np.ndarray,
    state_error_m: np.ndarray,
    P_m: np.ndarray,
    Pc_m: np.ndarray,
    outdir: Path,
    *,
    filename_prefix: str,
    title_prefix: str,
):
    t_hr = as_hours(t_s)
    sig2_p = 2.0 * np.sqrt(np.maximum(np.diagonal(P_m, axis1=1, axis2=2), 0.0))
    sig2_pc = 2.0 * np.sqrt(np.maximum(np.diagonal(Pc_m, axis1=1, axis2=2), 0.0))

    def make_panel(indices, labels, ylabel_unit, filename, title):
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        for row, k in enumerate(indices):
            ax[row].plot(t_hr, state_error_m[:, k], ".", color="k", markersize=2, label=r"$\Delta\hat{x}_k$")
            ax[row].plot(t_hr, sig2_p[:, k], "r", linewidth=1.0, label=r"$+2\sigma$ from $P_i$")
            ax[row].plot(t_hr, -sig2_p[:, k], "r", linewidth=1.0)
            ax[row].plot(t_hr, sig2_pc[:, k], "b--", linewidth=1.0, label=r"$+2\sigma$ from $P_{c,i}$")
            ax[row].plot(t_hr, -sig2_pc[:, k], "b--", linewidth=1.0)
            ax[row].set_ylabel(f"{labels[row]} [{ylabel_unit}]")
            if row == 0:
                ax[row].legend(loc="upper right", fontsize=8)

        ax[0].set_title(title)
        ax[-1].set_xlabel("Time [hours]")
        savefig(fig, outdir / f"{filename_prefix}_{filename}.png")

    make_panel([0, 1, 2], ["x", "y", "z"], "m", "pos", f"{title_prefix}: Position Error with $\\pm 2\\sigma$")
    make_panel([3, 4, 5], ["vx", "vy", "vz"], "m/s", "vel", f"{title_prefix}: Velocity Error with $\\pm 2\\sigma$")


def make_required_prob1b_plots(
    *,
    t_s: np.ndarray,
    state_error_m: np.ndarray,
    P_m: np.ndarray,
    Pc_m: np.ndarray,
    outdir: Path,
    zoom_t0_s: float,
):
    _plot_state_error_with_two_sigma(
        t_s=t_s,
        state_error_m=state_error_m,
        P_m=P_m,
        Pc_m=Pc_m,
        outdir=outdir,
        filename_prefix="state_error_two_sigma",
        title_prefix="CC (all times)",
    )

    zoom_mask = np.asarray(t_s, dtype=float) >= float(zoom_t0_s)
    if np.any(zoom_mask):
        _plot_state_error_with_two_sigma(
            t_s=t_s[zoom_mask],
            state_error_m=state_error_m[zoom_mask, :],
            P_m=P_m[zoom_mask, :, :],
            Pc_m=Pc_m[zoom_mask, :, :],
            outdir=outdir,
            filename_prefix=f"state_error_two_sigma_zoom_t_ge_{int(zoom_t0_s)}",
            title_prefix=f"CC (t >= {int(zoom_t0_s)} s)",
        )


def rms_summary(state_error_m: np.ndarray, t_s: np.ndarray, first_pass_gap_s: float = 6 * 3600.0) -> pd.DataFrame:
    e = np.asarray(state_error_m, dtype=float)
    t_s = np.asarray(t_s, dtype=float)

    mask_all = np.ones_like(t_s, dtype=bool)
    mask_ignore = t_s >= float(first_pass_gap_s)

    def comp_rms(mask):
        comp = np.sqrt(np.nanmean(e[mask, :] ** 2, axis=0))
        pos3 = np.sqrt(np.nanmean(np.sum(e[mask, 0:3] ** 2, axis=1)))
        vel3 = np.sqrt(np.nanmean(np.sum(e[mask, 3:6] ** 2, axis=1)))
        return comp, pos3, vel3

    comp_all, pos3_all, vel3_all = comp_rms(mask_all)
    comp_ign, pos3_ign, vel3_ign = comp_rms(mask_ignore)

    rows = [
        {"metric": "x [m]", "all": comp_all[0], "ignore_first_pass": comp_ign[0]},
        {"metric": "y [m]", "all": comp_all[1], "ignore_first_pass": comp_ign[1]},
        {"metric": "z [m]", "all": comp_all[2], "ignore_first_pass": comp_ign[2]},
        {"metric": "vx [m/s]", "all": comp_all[3], "ignore_first_pass": comp_ign[3]},
        {"metric": "vy [m/s]", "all": comp_all[4], "ignore_first_pass": comp_ign[4]},
        {"metric": "vz [m/s]", "all": comp_all[5], "ignore_first_pass": comp_ign[5]},
        {"metric": "||r||_2 [m]", "all": pos3_all, "ignore_first_pass": pos3_ign},
        {"metric": "||v||_2 [m/s]", "all": vel3_all, "ignore_first_pass": vel3_ign},
    ]
    return pd.DataFrame(rows)


def main():
    num_meas = None
    zoom_t0_s = 30000.0

    all_meas, truth_times, truth_states, stations, R, X0_star = load_hw2_setup(num_meas=num_meas)

    mu = 398600.4415
    J2 = 0.0010826269
    J3_truth = -2.5324e-6
    J3_filter = 0.0
    Re = 6378.0

    # Assignment setup (part a): P_xx,0 = 1e4 I.
    P0 = 1.0e4 * np.eye(6)
    outdir = Path(__file__).resolve().parent / "Plots" / "prob1b"
    outdir.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "case_id": "j3_considered",
            "label": "J3 considered (P_cc0 = J3^2)",
            "P_cc0": np.array([[J3_truth**2]], dtype=float),
        },
        {
            "case_id": "j3_not_considered",
            "label": "J3 not considered (P_cc0 = 0)",
            "P_cc0": np.array([[0.0]], dtype=float),
        },
    ]

    rms_all = []
    compare_curves = []

    for cfg in cases:
        out_cc = sequential_consider_cov_analysis(
            X0_star=X0_star,
            all_meas=all_meas,
            stations=stations,
            R=R,
            mu=mu,
            J2=J2,
            J3=J3_filter,
            Re=Re,
            P0=P0,
            P_cc0=cfg["P_cc0"],
            reltol=1.0e-10,
            abstol=1.0e-10,
            method="DOP853",
        )

        t_out = np.asarray(out_cc["t_meas"], dtype=float)
        station_meas = [m["station"] for m in all_meas[1 : 1 + t_out.size]]

        plot_data = convert_cc_outputs_to_plot_units(
            out_cc,
            truth_times=truth_times,
            truth_states_km=truth_states,
            station_meas=station_meas,
            R_km=R,
        )

        case_dir = outdir / cfg["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)

        make_postfit_residuals_linear_plot(plot_data["p_plot"], case_dir)
        make_trace_cov_pos_vel_plot(
            plot_data["p_plot"],
            case_dir,
            filename="trace_cov_pos_vel_Pi.png",
            title=f"{cfg['label']}: trace(P_i)",
        )
        make_trace_cov_pos_vel_plot(
            plot_data["pc_plot"],
            case_dir,
            filename="trace_cov_pos_vel_Pci.png",
            title=f"{cfg['label']}: trace(P_c,i)",
        )
        make_required_prob1b_plots(
            t_s=plot_data["t_meas"],
            state_error_m=plot_data["state_error_m"],
            P_m=plot_data["P_m"],
            Pc_m=plot_data["Pc_m"],
            outdir=case_dir,
            zoom_t0_s=zoom_t0_s,
        )

        rms_df = rms_summary(plot_data["state_error_m"], plot_data["t_meas"])
        rms_df.insert(0, "case", cfg["label"])
        rms_df.to_csv(case_dir / "state_error_rms_summary.csv", index=False)
        rms_all.append(rms_df)

        compare_curves.append(
            {
                "label": cfg["label"],
                "t_s": plot_data["t_meas"],
                "state_error_m": plot_data["state_error_m"],
            }
        )

        with pd.option_context("display.float_format", lambda x: f"{x:.6e}"):
            print(f"\nRMS summary ({cfg['label']}) [units: m, m/s]:")
            print(rms_df.drop(columns=["case"]).to_string(index=False))
        print(f"Saved case outputs to: {case_dir}")

    rms_cmp = pd.concat(rms_all, ignore_index=True)
    rms_cmp_csv = outdir / "rms_comparison.csv"
    rms_cmp.to_csv(rms_cmp_csv, index=False)

    # Quick visual compare of 3D state-error norms.
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for item in compare_curves:
        t_hr = as_hours(item["t_s"])
        e = item["state_error_m"]
        pos_norm = np.linalg.norm(e[:, 0:3], axis=1)
        vel_norm = np.linalg.norm(e[:, 3:6], axis=1)
        ax[0].plot(t_hr, pos_norm, ".", markersize=2, label=item["label"])
        ax[1].plot(t_hr, vel_norm, ".", markersize=2, label=item["label"])

    ax[0].set_ylabel(r"$||e_r||_2$ [m]")
    ax[0].set_title("1b Case Comparison: Position Error Norm")
    ax[1].set_ylabel(r"$||e_v||_2$ [m/s]")
    ax[1].set_title("1b Case Comparison: Velocity Error Norm")
    ax[1].set_xlabel("Time [hours]")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].legend(loc="upper right", fontsize=8)
    savefig(fig, outdir / "state_error_norm_comparison.png")

    print(f"\nSaved comparison outputs under: {outdir}")
    print(f"RMS comparison CSV: {rms_cmp_csv}")


if __name__ == "__main__":
    main()
