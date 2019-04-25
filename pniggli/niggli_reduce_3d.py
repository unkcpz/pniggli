from typing import Tuple

import numpy as np

__all__ = ['niggli_reduce', 'niggli_check']

reduced_lattice = np.ndarray

def niggli_reduce(lattice, eps: float=1e-5, loop_max=100) -> reduced_lattice:
    """
    niggli reduction

    :param lattice: Origin lattice
    :type lattice: list or np.ndarray with 9 elements
    :param eps: tolerance
    :type eps: float default 1e-5
    :return: a reduced lattice
    :rtype: 3x3 np.ndarray
    """
    try:
        reduced_lattice = np.array(lattice).reshape((3, 3))
    except:
        raise ValueError("Must 9 elements for 3x3 lattice")

    L = reduced_lattice
    G = _get_metric(L)

    # This sets an upper limit on the number of iterations.
    reduced = False
    count = 0
    while not reduced and count <= loop_max:
        reduced = True
        count += 1
        # step 0: get parameters for A1-A8
        # X, E, Z for xi, eta, zeta respectively
        G = _get_metric(L)
        A, B, C, X, E, Z = _get_G_param(G)

        # step 1
        if A > B + eps or (abs(A - B) < eps and abs(X) > abs(E) + eps):
            M = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
            L = np.matmul(L, M)
            reduced = False
        # step 2
        if (B > C + eps) or (abs(B - C) < eps and abs(E) > abs(Z) + eps):
            M = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            L = np.matmul(L, M)
            reduced = False
            continue

        l, m, n = _get_angle_param(X, E, Z, eps)
        # step 3
        if l * m * n == 1:
            i, j, k = l, m, n
            M = np.array([[i, 0, 0], [0, j, 0], [0, 0, k]])
            L = np.matmul(L, M)
            reduced = False
        # step 4
        elif l * m * n == 0 or l * m * n == -1:
            i = -1 if l == 1 else 1
            j = -1 if m == 1 else 1
            k = -1 if n == 1 else 1

            if i * j * k == -1:
                if n == 0:
                    k = -1
                elif m == 0:
                    j = -1
                elif l == 0:
                    i = -1
            M = np.array([[i, 0, 0], [0, j, 0], [0, 0, k]])
            L = np.matmul(L, M)
            reduced = False

        G = _get_metric(L)
        A, B, C, X, E, Z = _get_G_param(G)
        # step 5
        if (
            abs(X) > B + eps
            or (abs(X - B) < eps and 2 * E < Z - eps)
            or (abs(X + B) < eps and Z < -eps)
        ):
            M = np.array([[1, 0, 0], [0, 1, -np.sign(X)], [0, 0, 1]])
            L = np.matmul(L, M)
            reduced = False
            continue

        # step 6
        if (
            abs(E) > A + eps
            or (abs(A - E) < eps and 2 * X < Z - eps)
            or (abs(A + E) < eps and Z < -eps)
        ):
            M = np.array([[1, 0, -np.sign(E)], [0, 1, 0], [0, 0, 1]])
            L = np.matmul(L, M)
            reduced = False
            continue

        # step 7
        if (
            abs(Z) > A + eps
            or (abs(A - Z) < eps and 2 * X < E - eps)
            or (abs(A + Z) < eps and E < -eps)
        ):
            M = np.array([[1, -np.sign(Z), 0], [0, 1, 0], [0, 0, 1]])
            L = np.matmul(L, M)
            reduced = False
            continue

        # step 8
        if X + E + Z + A + B < -eps or (abs(X + E + Z + A + B) < eps < Z + (A + E) * 2):
            M = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
            L = np.matmul(L, M)
            reduced = False
            continue

    reduced_lattice = L
    return reduced_lattice

def _get_G_param(G, eps=1e-5) -> Tuple[float, float, float, float, float, float, Tuple[int]]:
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
    return np.matmul(M.T, M)

def niggli_check(L, eps=1e-5) -> bool:
    G = _get_metric(L)
    A, B, C, X, E, Z = _get_G_param(G)
    return _niggli_check(A, B, C, X, E, Z, eps)

def _niggli_check(A, B, C, X, E, Z, eps=1e-5):
    """Checks that the niggli reduced cell satisfies the niggli conditions.
    Conditions listed at: https://arxiv.org/pdf/1203.5146.pdf.
    Args:
        A (float): a.a
        B (float): b.b
        C (float): c.c
        xi (float): 2*b.c
        eta (float): 2*c.a
        zeta (float): 2*a.b

    Returns:
        False if niggli conditons aren't met.
    """
    if not (A-eps > 0 and (A < B-eps or np.allclose(A,B,atol=eps)) and
            (B < C-eps or np.allclose(B,C,atol=eps))):
        return False

    if np.allclose(A,B,atol=eps) and not (abs(X) < abs(E)-eps or
                                          np.allclose(abs(X),abs(E),atol=eps)):
        return False

    if np.allclose(B,C,atol=eps) and not (abs(E) < abs(Z)-eps
                                          or np.allclose(abs(E),abs(Z),atol=eps)):
        return False

    if not ((X-eps > 0 and E-eps > 0 and Z-eps > 0) or
            ((X < 0-eps or np.allclose(X,0,atol=eps))
             and (E < 0-eps or np.allclose(E,0,atol=eps))
             and (Z < 0-eps or np.allclose(Z,0,atol=eps)))):
        return False

    if not (abs(X) < B-eps or np.allclose(abs(X),B,atol=eps)):
        return False

    if not ((abs(E) < A-eps or np.allclose(abs(E),A,atol=eps)) and (abs(Z) < A-eps or
                                                           np.allclose(abs(Z),A,atol=eps))):
        return False

    if not (C < A+B+C+X+E+Z-eps or np.allclose(C, A+B+C+X+E+Z,atol=eps)):
        return False

    if np.allclose(X,B,atol=eps) and not (Z < 2.*E-eps or
                                           np.allclose(Z,2.*E,atol=eps)):
        return False

    if np.allclose(E,A,atol=eps) and not (Z < 2.*X-eps or
                                            np.allclose(Z,2.*X,atol=eps)):
        return False

    if np.allclose(Z,A,atol=eps) and not (E < 2.*X-eps or
                                             np.allclose(E,2.*X,atol=eps)):
        return False

    if np.allclose(X,-B,atol=eps) and not np.allclose(Z,0,atol=eps):
        return False

    if np.allclose(E,-A,atol=eps) and not np.allclose(Z,0,atol=eps):
        return False

    if np.allclose(Z,-A,atol=eps) and not np.allclose(E,0,atol=eps):
        return False

    if np.allclose(C,A+B+C+X+E+Z,rtol=0.0) and not ((2.*A+2.*E+Z) < 0-eps or
                                                 np.allclose(2.*A+2.*E+Z,0,atol=eps)):
        return False

    return True
