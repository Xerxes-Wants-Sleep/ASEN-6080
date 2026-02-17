import numpy as np


class Stations:
    def __init__(
            self,
            name: str,
            lat_deg: float,
            lon_deg: float,
            theta0_deg: float,
            radius_earth: float,
            w_earth_rad_per_s: float,
            use_ecef: bool = False,
            r_ecef: np.ndarray | None = None,
            elevation_mask_deg: float = 10,
            
        ):
    
        self.name = name
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.use_ecef = use_ecef
        self.elevation_mask_deg = float(elevation_mask_deg)
        self.radius_earth = float(radius_earth)
        self.theta0_deg = float(theta0_deg)
        self.w_earth_rad_per_s = float(w_earth_rad_per_s)
        self.elevation_mask_rad = float(np.deg2rad(self.elevation_mask_deg))
        self.theta0_rad = float(np.deg2rad(self.theta0_deg))
        self.lat_rad = float(np.deg2rad(self.lat_deg))
        self.lon_rad = float(np.deg2rad(self.lon_deg))


        if self.use_ecef:
            self.r_ecef = np.asarray(r_ecef, dtype=float).reshape(3)
            # lat/lon not used in this mode, but we keep them for compatibility
            self.lat_deg = float(lat_deg)
            self.lon_deg = float(lon_deg)
        else:
            self.lat_deg = float(lat_deg)
            self.lon_deg = float(lon_deg)
            self.r_ecef = self.lat_lon2ecef()



    def lat_lon2ecef(self) -> np.ndarray:
        ''''''
        x = self.radius_earth * np.cos(self.lat_rad) * np.cos(self.lon_rad)
        y = self.radius_earth * np.cos(self.lat_rad) * np.sin(self.lon_rad)
        z = self.radius_earth * np.sin(self.lat_rad)
        return np.array([x, y, z], dtype=float)
    

    def ecef2eci(self, t: float, r_sc_ecef: np.ndarray, v_sc_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = self.theta0_rad + self.w_earth_rad_per_s*t
        cos: float = np.cos(theta)
        sin: float = np.sin(theta)

        R3: np.ndarray = np.array([[cos, -sin, 0],
                                [sin, cos, 0],
                                [ 0,   0,   1]])
        
        r_eci = R3 @ r_sc_ecef
        v_eci = R3 @ v_sc_ecef + np.cross(np.array([0.0, 0.0, self.w_earth_rad_per_s]), r_eci)
        X_eci = np.hstack([r_eci, v_eci])

        return r_eci, v_eci, X_eci
    
    def station_eci(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Station position/velocity in ECI at time t."""
        v_zero = np.zeros(3)
        r_st, v_st, _ = self.ecef2eci(t, self.r_ecef, v_zero)
        return r_st, v_st

    def elevation(self, t: float, r_sc_eci: np.ndarray) -> float:

        r_st, _ = self.station_eci(t)
        rho_vec: np.ndarray = r_sc_eci - r_st
        rho_hat: np.ndarray = rho_vec / np.linalg.norm(rho_vec)
        up: np.ndarray = r_st / np.linalg.norm(r_st)
        elev = np.arcsin(np.dot(rho_hat, up))
        return float(elev)
    

    def measure(self, r_sc_eci: np.ndarray, v_sc_eci: np.ndarray, t: float) -> dict | None:

        # Compute station ECI once and reuse for elevation check + measurement
        r_st, v_st = self.station_eci(t)

        rho_vec_elev = r_sc_eci - r_st
        rho_hat_elev = rho_vec_elev / np.linalg.norm(rho_vec_elev)
        up = r_st / np.linalg.norm(r_st)
        elev = float(np.arcsin(np.dot(rho_hat_elev, up)))

        if elev < self.elevation_mask_rad:
            return None

        rho_vec = rho_vec_elev  # already computed above
        rho = np.linalg.norm(rho_vec)
        rho_hat = rho_vec / rho

        rho_dot = np.dot(rho_hat, (v_sc_eci - v_st))

        return {
        "station": self.name,
        "t": t,
        # Keep both key styles for compatibility across HW2/HW3 and project code.
        "rho": float(rho),
        "rho_dot": float(rho_dot),
        "rho_km": float(rho),
        "rho_dot_km_s": float(rho_dot),
        "elev_rad": elev
        }