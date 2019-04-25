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
    # import pdb; pdb.set_trace()

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
        # import pdb; pdb.set_trace()

        # step 1
        if A > B + eps or (abs(A - B) < eps and abs(X) > abs(E) + eps):
            M = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
            # G = np.dot(M.T, np.dot(G, M))
            # L = np.matmul(M.T, L)
            L = np.matmul(L, M)
            reduced = False
        # step 2
        if (B > C + eps) or (abs(B - C) < eps and abs(E) > abs(Z) + eps):
            M = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            # G = np.dot(M.T, np.dot(G, M))
            # L = np.matmul(M.T, L)
            L = np.matmul(L, M)
            reduced = False
            continue

        l, m, n = _get_angle_param(X, E, Z, eps)
        # step 3
        if l * m * n == 1:
            # i = -1 if l == -1 else 1
            # j = -1 if m == -1 else 1
            # k = -1 if n == -1 else 1
            i, j, k = l, m, n
            M = np.array([[i, 0, 0], [0, j, 0], [0, 0, k]])
            # G = np.dot(M.T, np.dot(G, M))
            # L = np.matmul(M.T, L)
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
            # G = np.dot(M.T, np.dot(G, M))
            # L = np.matmul(M.T, L)
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
            # G = np.dot(M.T, np.dot(G, M))
            # L = np.matmul(M.T, L)
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
            # G = np.dot(M.T, np.dot(G, M))
            # L = np.matmul(M.T, L)
            L = np.matmul(L, M)
            reduced = False
            continue

        # step 8
        if X + E + Z + A + B < -eps or (abs(X + E + Z + A + B) < eps < Z + (A + E) * 2):
            M = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
            # G = np.dot(M.T, np.dot(G, M))
            # L = np.matmul(M.T, L)
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
    xi = 2 * G[1, 2]
    eta = 2 * G[0, 2]
    zeta = 2 * G[0, 1]

    return A, B, C, xi, eta, zeta

def _get_angle_param(xi, eta, zeta, eps=1e-5) -> Tuple[int, int, int]:
    # angle types
    l = _get_angle_type(xi, eps)
    m = _get_angle_type(eta, eps)
    n = _get_angle_type(zeta, eps)

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

def niggli_check(L, eps):
    G = _get_metric(L)
    A, B, C, X, E, Z = _get_G_param(G)
    return _niggli_check(A, B, C, X, E, Z, eps)

def _niggli_check(A,B,C,xi,eta,zeta,eps):
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

    # import pdb; pdb.set_trace()

    if not (A-eps > 0 and (A < B-eps or np.allclose(A,B,atol=eps)) and
            (B < C-eps or np.allclose(B,C,atol=eps))):
        return False

    if np.allclose(A,B,atol=eps) and not (abs(xi) < abs(eta)-eps or
                                          np.allclose(abs(xi),abs(eta),atol=eps)):
        return False

    if np.allclose(B,C,atol=eps) and not (abs(eta) < abs(zeta)-eps
                                          or np.allclose(abs(eta),abs(zeta),atol=eps)):
        return False

    if not ((xi-eps > 0 and eta-eps > 0 and zeta-eps > 0) or
            ((xi < 0-eps or np.allclose(xi,0,atol=eps))
             and (eta < 0-eps or np.allclose(eta,0,atol=eps))
             and (zeta < 0-eps or np.allclose(zeta,0,atol=eps)))):
        return False

    if not (abs(xi) < B-eps or np.allclose(abs(xi),B,atol=eps)):
        return False

    if not ((abs(eta) < A-eps or np.allclose(abs(eta),A,atol=eps)) and (abs(zeta) < A-eps or
                                                           np.allclose(abs(zeta),A,atol=eps))):
        return False

    if not (C < A+B+C+xi+eta+zeta-eps or np.allclose(C, A+B+C+xi+eta+zeta,atol=eps)):
        return False

    if np.allclose(xi,B,atol=eps) and not (zeta < 2.*eta-eps or
                                           np.allclose(zeta,2.*eta,atol=eps)):
        return False

    if np.allclose(eta,A,atol=eps) and not (zeta < 2.*xi-eps or
                                            np.allclose(zeta,2.*xi,atol=eps)):
        return False

    if np.allclose(zeta,A,atol=eps) and not (eta < 2.*xi-eps or
                                             np.allclose(eta,2.*xi,atol=eps)):
        return False

    if np.allclose(xi,-B,atol=eps) and not np.allclose(zeta,0,atol=eps):
        return False

    if np.allclose(eta,-A,atol=eps) and not np.allclose(zeta,0,atol=eps):
        return False

    if np.allclose(zeta,-A,atol=eps) and not np.allclose(eta,0,atol=eps):
        return False

    if np.allclose(C,A+B+C+xi+eta+zeta,rtol=0.0) and not ((2.*A+2.*eta+zeta) < 0-eps or
                                                 np.allclose(2.*A+2.*eta+zeta,0,atol=eps)):
        return False

    return True
