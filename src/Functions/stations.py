import numpy as np


class Stations:
    def __init__(
            self,
            name: str,
            lat_deg: float,
            lon_deg: float,
            elevation_mask_deg: float = 10,
            radius_earth: float = 6378,
            theta0_deg: float = 122,
            w_earth_rad_per_s: float =2*np.pi/(24*60*60)
        ):
    
        self.name = name
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.elevation_mask_deg = float(elevation_mask_deg)
        self.radius_earth = float(radius_earth)
        self.theta0_deg = float(theta0_deg)
        self.w_earth_rad_per_s = float(w_earth_rad_per_s)
        self.deg2radians()
        self.r_ecef = self.lat_lon2ecef()



    def deg2radians(self) -> None:
        self.lat_rad:float = np.deg2rad(self.lat_deg)
        self.lon_rad:float = np.deg2rad(self.lon_deg)
        self.elevation_mask_rad: float = np.deg2rad(self.elevation_mask_deg)
        self.theta0_rad:float = np.deg2rad(self.theta0_deg)

    

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

    def elevation(self, t: float, r_sc_eci: np.ndarray) -> float:

        v_zero = np.zeros(3)
        r_st, _, _ = self.ecef2eci(t, self.r_ecef, v_zero)
        rho_vec: np.ndarray = r_sc_eci - r_st
        rho_hat: np.ndarray = rho_vec / np.linalg.norm(rho_vec)
        up: np.ndarray = r_st / np.linalg.norm(r_st)
        elev = np.arcsin(np.dot(rho_hat, up))
        return float(elev)
    

    def measure(self, r_sc_eci: np.ndarray, v_sc_eci: np.ndarray, t: float) -> dict | None:

        elev = self.elevation(t, r_sc_eci)
        
        if elev < self.elevation_mask_rad:
            return None
        
        v_zero = np.zeros(3)
        r_st, v_st, _ = self.ecef2eci(t, self.r_ecef, v_zero)

        rho_vec = r_sc_eci - r_st
        rho = np.linalg.norm(rho_vec)
        rho_hat = rho_vec / rho

        rho_dot = np.dot(rho_hat, (v_sc_eci - v_st))

        return {
        "station": self.name,
        "t": t,
        "rho_km": rho,
        "rho_dot_km_s": rho_dot,
        "elev_rad": elev
        }








    


        
        




