import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("../../")

from src.Functions.stations import Stations


def build_stations() -> list[Stations]:
    theta0_deg = 122.0
    w_earth_rad_per_s = 2.0 * np.pi / (24.0 * 3600.0)
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


def generate_measurements(
    *,
    truth_path: Path,
    out_path_clean: Path,
    out_path_noisy: Path,
    sigma_rho_km: float,
    sigma_rhod_km_s: float,
    seed: int,
) -> None:
    truth = pd.read_csv(truth_path)

    required_cols = {"t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"}
    missing = required_cols.difference(truth.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Truth CSV is missing required columns: {missing_sorted}")

    t = truth["t_s"].to_numpy(float)
    r = truth[["x_km", "y_km", "z_km"]].to_numpy(float)
    v = truth[["vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)

    stations = build_stations()

    all_meas = []
    for k in range(len(t)):
        for st in stations:
            m = st.measure(r[k], v[k], float(t[k]))
            if m is not None:
                all_meas.append(m)

    if not all_meas:
        raise RuntimeError("No visible measurements produced. Check truth data and station mask settings.")

    df = pd.DataFrame(all_meas)  # station, t, rho_km, rho_dot_km_s, elev_rad

    out_path_clean.parent.mkdir(parents=True, exist_ok=True)
    out_path_noisy.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path_clean, index=False)
    print("Saved clean:", out_path_clean)
    print("First/last measurement epochs [s]:", float(df["t"].iloc[0]), float(df["t"].iloc[-1]))
    print("Measurements:", len(df))

    rng = np.random.default_rng(seed)
    df_noisy = df.copy()
    df_noisy["rho_km"] = df_noisy["rho_km"] + rng.normal(0.0, sigma_rho_km, size=len(df_noisy))
    df_noisy["rho_dot_km_s"] = df_noisy["rho_dot_km_s"] + rng.normal(
        0.0, sigma_rhod_km_s, size=len(df_noisy)
    )

    df_noisy.to_csv(out_path_noisy, index=False)
    print("Saved noisy:", out_path_noisy)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Generate clean and noisy range/range-rate measurements from a truth trajectory CSV."
    )
    parser.add_argument(
        "--truth-path",
        type=Path,
        default=repo_root / "Homework" / "Homework1" / "HW2_j3_on_truth.csv",
        help="Path to truth trajectory CSV with columns t_s, x_km, y_km, z_km, vx_km_s, vy_km_s, vz_km_s.",
    )
    parser.add_argument(
        "--out-clean",
        type=Path,
        default=repo_root / "Homework" / "Homework2" / "meas_data" / "prob2_hw2_measurements_clean.csv",
        help="Output path for clean measurements CSV.",
    )
    parser.add_argument(
        "--out-noisy",
        type=Path,
        default=repo_root / "Homework" / "Homework2" / "meas_data" / "prob2_hw2_measurements_noisy.csv",
        help="Output path for noisy measurements CSV.",
    )
    parser.add_argument("--sigma-rho-km", type=float, default=1e-3, help="Range noise sigma [km].")
    parser.add_argument("--sigma-rhod-km-s", type=float, default=1e-6, help="Range-rate noise sigma [km/s].")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for repeatable noise.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_measurements(
        truth_path=args.truth_path,
        out_path_clean=args.out_clean,
        out_path_noisy=args.out_noisy,
        sigma_rho_km=float(args.sigma_rho_km),
        sigma_rhod_km_s=float(args.sigma_rhod_km_s),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()

