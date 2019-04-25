from typing import Tuple
import numpy as np

from .utils_2d import _get_G_param, _get_metric, _get_angle_type, _get_angle_param

__all__ = ['niggli_reduce_2d']

reduced_lattice = np.ndarray

def niggli_reduce_2d(lattice, eps: float=1e-5, loop_max=100) -> reduced_lattice:
    """
    niggli reduction

    :param lattice: Origin lattice
    :type lattice: list or np.ndarray with 4 elements
    :param eps: tolerance
    :type eps: float default 1e-5
    :return: a reduced lattice
    :rtype: 2x2 np.ndarray
    """
    try:
        reduced_lattice = np.array(lattice).reshape((2, 2))
    except:
        raise ValueError("Must 4 elements for 2x2 lattice")

    L = reduced_lattice
    G = _get_metric(L)

    for _ in range(loop_max):
        reduced = True

        # step 0: get parameters for A1-A8
        # Y for gamma respectively
        G = _get_metric(L)
        A, B, Y = _get_G_param(G)

        # step 1
        if A > B + eps:
            M = np.array([[0, 1],
                          [1, 0]])
            L = np.matmul(M, L)
            reduced = False
            continue

        # step 2
        # if acute, transform to obtuse
        if Y > 0:
            M = np.array([[1, 0],
                          [0, -1]])
            L = np.matmul(M, L)
            reduced = False
            continue

        # step 3
        if abs(Y) > A + eps:
            M = np.array([[1, 0],
                          [-np.sign(Y), 1]])
            L = np.matmul(M, L)
            reduced = False
            continue

        # step 4
        if abs(Y) > B + eps:
            M = np.array([[1, -np.sign(Y)],
                          [0, 1]])
            L = np.matmul(M, L)
            reduced = False
            continue

        if reduced:
            break

    reduced_lattice = L
    return reduced_lattice
