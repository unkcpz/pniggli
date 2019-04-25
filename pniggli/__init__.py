from .reduce_3d import niggli_reduce_3d, niggli_check
from .reduce_2d import niggli_reduce_2d

import numpy as np


__all__ = ['niggli_reduce', 'niggli_check']

reduced_lattice = np.ndarray

def niggli_reduce(lattice, eps: float=1e-5, loop_max=100) -> reduced_lattice:
    """
    niggli reduction

    :param lattice: Origin lattice
    :type lattice: list or np.ndarray with 4 or 9 elements
    :param eps: tolerance
    :type eps: float default 1e-5
    :return: a reduced lattice
    :rtype: 2x2 or 3x3 np.ndarray
    """
    L = np.array(lattice)
    if L.size == 4:
        return niggli_reduce_2d(L)
    if L.size == 9:
        return niggli_reduce_3d(L)

    raise ValueError("Must a 2x2 or 3x3 lattice")
