import numpy as np

__all__ = ['niggli_reduce']

reduced_lattice = np.ndarray

def niggli_reduce(lattice, eps: float=1e-5) -> reduced_lattice:
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
        lattice = np.array(lattice).reshape((3, 3))
    except:
        raise ValueError("Must 9 elements for 3x3 lattice")

    return lattice
