import numpy as np

def SOIcheck(t: float, X: np.ndarray) -> float:
    """
    MATLAB-style SOI event value:
      value = r - 3*RSOI
    """
    RSOI = 925000.0
    x, y, z = X[0], X[1], X[2]
    r = float(np.sqrt(x * x + y * y + z * z))
    return float(r - 3.0 * RSOI)