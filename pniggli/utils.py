from typing import Tuple
import numpy as np

def _get_G_param(G, eps=1e-5) -> Tuple[float, float, float, float, float, float]:
    """
    A = a.a = G[0, 0]
    B = b.b = G[1, 1]
    C = c.c = G[2, 2]
    xi = 2.b.c = 2 * G[1, 2]
    eta = 2.c.a = 2 * G[0, 2]
    zeta = 2.a.b = 2 * G[0, 1]
    """
    A = G[0, 0]
    B = G[1, 1]
    C = G[2, 2]
    X = 2 * G[1, 2]
    E = 2 * G[0, 2]
    Z = 2 * G[0, 1]

    return A, B, C, X, E, Z

def _get_angle_param(X, E, Z, eps=1e-5) -> Tuple[int, int, int]:
    # angle types
    l = _get_angle_type(X, eps)
    m = _get_angle_type(E, eps)
    n = _get_angle_type(Z, eps)

    return l, m, n


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
    M = lattice.reshape((3, 3))
    return np.matmul(M, M.T)
