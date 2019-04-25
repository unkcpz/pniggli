import numpy as np
from typing import Tuple

def _get_angle_type(angle, eps=1e-5) -> int:
    """
    -------------
    Angle	value
    -----   -----
    Acute	1
    Obtuse	-1
    Right	0
    -------------
    """
    if angle < -eps:
        return -1
    elif angle > eps:
        return 1
    else: # -eps < angle < eps
        return 0

def _get_metric(lattice) -> np.ndarray:
    M = lattice.reshape((2, 2))
    return np.matmul(M, M.T)

def _get_G_param(G, eps=1e-5) -> Tuple[float, float, float]:
    """
    A = a.a = G[0, 0]
    B = b.b = G[1, 1]
    Y = 2.a.b = 2 * G[0, 1]
    """
    A = G[0, 0]
    B = G[1, 1]
    Y = 2 * G[0, 1]

    return A, B, Y

def _get_angle_param(Y, eps=1e-5) -> int:
    # angle types
    l = _get_angle_type(Y, eps)

    return l
