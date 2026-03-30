import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

sys.path.append("../../")

from src.Functions.consider_cov_filter import sequential_consider_cov_analysis
from src.Functions.jacobians import orbit_propagator
from src.Functions.stations import Stations
from src.helpers.plotting.common import as_hours, savefig


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

    # Match the same setup used in Homework 2 and prob_1b.
    dx0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)*0
    X0_star = x0_true + dx0

    sigma_rho_km = 1.0e-3
    sigma_rhod_km_s = 1.0e-6
    R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

    stations = build_stations()

    return all_meas, truth_times, truth_states, stations, R, X0_star


def interp_truth_at_times(t_s: np.ndarray, truth_times: np.ndarray, truth_states: np.ndarray) -> np.ndarray:
    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    return np.column_stack([np.interp(t_s, truth_times, truth_states[:, i]) for i in range(6)])


def propagate_state_history(
    *,
    X0_km: np.ndarray,
    t_eval: np.ndarray,
    mu: float,
    J2: float,
    J3_filter: float,
    Re: float,
    reltol: float = 1.0e-10,
    abstol: float = 1.0e-10,
    method: str = "DOP853",
) -> np.ndarray:
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    
    sol = solve_ivp(
        fun=lambda t, y: orbit_propagator(t, y, mu, J2, J3_filter, Re=Re, j2=True, j3=False),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=np.asarray(X0_km, dtype=float).reshape(6,),
        t_eval=t_eval,
        rtol=reltol,
        atol=abstol,
        method=method,
    )
    if not sol.success:
        raise RuntimeError(f"State propagation failed: {sol.message}")
    return sol.y.T


def map_final_to_epoch_and_forward(
    *,
    out_cc: dict,
    P_cc0: np.ndarray,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Final sequential outputs (at t_f).
    # Part (d) asks for mapping Delta x_f^+ and P_c,f^+.
    x_f_plus = np.asarray(out_cc["xEst"], dtype=float)[-1, :]
    Pc_f_plus = np.asarray(out_cc["PcEst"], dtype=float)[-1, :, :]
    Pxc_f_plus = np.asarray(out_cc["PxcEst"], dtype=float)[-1, :, :]
    Psi_hist = np.asarray(out_cc["Psi"], dtype=float)  # (m, 7, 7), m = num_output_epochs

    q = np.asarray(P_cc0, dtype=float).shape[0]
    
    Psi_f0 = Psi_hist[-1, :, :]

    P_aug_f_plus = np.block(
        [
            [Pc_f_plus, Pxc_f_plus],
            [Pxc_f_plus.T, np.asarray(P_cc0, dtype=float)],
        ]
    )

    # Consider-parameter estimate is not updated in this filter, so c_hat_f = 0.
    z_f_plus = np.hstack((x_f_plus, np.zeros(q)))

    # Back-map to epoch t0.
    z0_plus = np.linalg.solve(Psi_f0, z_f_plus)

    # Correct: Psi^{-1} P Psi^{-T}
    Psi_inv = np.linalg.inv(Psi_f0)
    P_aug_0_plus = Psi_inv @ P_aug_f_plus @ Psi_inv.T
    P_aug_0_plus = 0.5 * (P_aug_0_plus + P_aug_0_plus.T)

    # Forward-map to each output epoch with all-l information.
    m = Psi_hist.shape[0]
    P_aug_il = np.zeros_like(Psi_hist)
    for k in range(m):
        Psi_k0 = Psi_hist[k, :, :]
        P_aug_il[k, :, :] = Psi_k0 @ P_aug_0_plus @ Psi_k0.T
        P_aug_il[k, :, :] = 0.5 * (P_aug_il[k, :, :] + P_aug_il[k, :, :].T)

    return z0_plus, P_aug_0_plus, P_aug_il


def plot_state_errors_with_2sigma(
    *,
    t_s: np.ndarray,
    state_error_m: np.ndarray,
    P_m: np.ndarray,
    outdir: Path,
    filename_prefix: str,
    title_prefix: str,
):
    t_hr = as_hours(t_s)
    sig2 = 2.0 * np.sqrt(np.maximum(np.diagonal(P_m, axis1=1, axis2=2), 0.0))

    def make_panel(indices, labels, ylabel_unit, filename, title):
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        for row, k in enumerate(indices):
            ax[row].plot(t_hr, state_error_m[:, k], ".", color="k", markersize=2, label="state error")
            ax[row].plot(t_hr, sig2[:, k], "r", linewidth=1.0, label=r"$+2\sigma$ envelope")
            ax[row].plot(t_hr, -sig2[:, k], "r", linewidth=1.0)
            ax[row].set_ylabel(f"{labels[row]} [{ylabel_unit}]")
            if row == 0:
                ax[row].legend(loc="upper right", fontsize=8)

        ax[0].set_title(title)
        ax[-1].set_xlabel("Time [hours]")
        savefig(fig, outdir / f"{filename_prefix}_{filename}.png")

    make_panel(
        [0, 1, 2],
        ["x", "y", "z"],
        "m",
        "pos",
        f"{title_prefix}: Position Error with $\\pm 2\\sigma$",
    )
    make_panel(
        [3, 4, 5],
        ["vx", "vy", "vz"],
        "m/s",
        "vel",
        f"{title_prefix}: Velocity Error with $\\pm 2\\sigma$",
    )


def make_prob1d_plots(
    *,
    t_s: np.ndarray,
    state_error_m: np.ndarray,
    Pci_il_m: np.ndarray,
    outdir: Path,
    zoom_t0_s: float = 30000.0,
):
    plot_state_errors_with_2sigma(
        t_s=t_s,
        state_error_m=state_error_m,
        P_m=Pci_il_m,
        outdir=outdir,
        filename_prefix="state_error_epoch_mapped",
        title_prefix="Part d",
    )

    mask_zoom = np.asarray(t_s, dtype=float) >= float(zoom_t0_s)
    if np.any(mask_zoom):
        plot_state_errors_with_2sigma(
            t_s=t_s[mask_zoom],
            state_error_m=state_error_m[mask_zoom, :],
            P_m=Pci_il_m[mask_zoom, :, :],
            outdir=outdir,
            filename_prefix=f"state_error_epoch_mapped_zoom_t_ge_{int(zoom_t0_s)}",
            title_prefix=f"Part d Zoomed",
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

    # Same assignment setup as part b.
    P0 = 1.0e4 * np.eye(6)
    P_cc0 = np.array([[J3_truth**2]], dtype=float)

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
        P_cc0=P_cc0,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
    )

    # Output epochs correspond to t_1 ... t_f
    t_il = np.asarray(out_cc["t_meas"], dtype=float)

    # Map final + quantities back to t0, then map covariance forward as i|l
    z0_plus, P_aug_0_plus, P_aug_il = map_final_to_epoch_and_forward(out_cc=out_cc, P_cc0=P_cc0)
    dx0_plus = z0_plus[:6]
    X0_hat_epoch = np.asarray(X0_star, dtype=float).reshape(6,) + dx0_plus

    # Propagate new epoch estimate forward to the same output epochs.
    t0 = float(all_meas[0]["t"])
    t_eval_all = np.hstack(([t0], t_il))
    X_epoch_hist_all = propagate_state_history(
        X0_km=X0_hat_epoch,
        t_eval=t_eval_all,
        mu=mu,
        J2=J2,
        J3_filter=J3_filter,
        Re=Re,
        reltol=1.0e-10,
        abstol=1.0e-10,
        method="DOP853",
    )
    X_epoch_hist = X_epoch_hist_all[1:, :]

    # Truth and state errors at t_i (i = 1..f).
    Xtrue_il = interp_truth_at_times(t_il, truth_times, truth_states)
    state_error_il_m = (X_epoch_hist - Xtrue_il) * 1000.0

    # Extract mapped i|l consider covariance in state block.
    Pci_il = P_aug_il[:, :6, :6]
    Pci_il_m = Pci_il * (1000.0**2)

    outdir = Path(__file__).resolve().parent / "Plots" / "prob1d_no_offset"
    outdir.mkdir(parents=True, exist_ok=True)

    make_prob1d_plots(
        t_s=t_il,
        state_error_m=state_error_il_m,
        Pci_il_m=Pci_il_m,
        outdir=outdir,
        zoom_t0_s=zoom_t0_s,
    )

    rms_df = rms_summary(state_error_il_m, t_il)
    rms_csv = outdir / "state_error_rms_summary_epoch_mapped.csv"
    rms_df.to_csv(rms_csv, index=False)

    epoch_df = pd.DataFrame(
        {
            "component": ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"],
            "X0_star": np.asarray(X0_star, dtype=float),
            "delta_x0_plus": dx0_plus,
            "X0_hat_epoch": X0_hat_epoch,
        }
    )
    epoch_csv = outdir / "epoch_estimate_from_backmap.csv"
    epoch_df.to_csv(epoch_csv, index=False)

    with pd.option_context("display.float_format", lambda x: f"{x:.6e}"):
        print("\nMapped epoch estimate summary (km, km/s):")
        print(epoch_df.to_string(index=False))
        print("\nPart d RMS summary (units: m, m/s):")
        print(rms_df.to_string(index=False))

    print(f"\nSaved part (d) outputs under: {outdir}")
    print(f"Epoch CSV: {epoch_csv}")
    print(f"RMS CSV:   {rms_csv}")

    # Keep these values visible in logs for quick sanity checks.
    print("\nBack-mapped covariance blocks at epoch (km-based units):")
    print("P_c,0|l (state block):")
    print(P_aug_0_plus[:6, :6])
    print("P_xc,0|l (cross block):")
    print(P_aug_0_plus[:6, 6:])


if __name__ == "__main__":
    main()
