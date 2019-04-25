pniggli
========================================

This is a (P)ure python implementation of algorithm to determin Niggli cell.
The library supports both 2D and 3D niggli transformations.

Rows of list or rows of `numpy.ndarray` correspond basis vectors, a, b, c or a, b
They are input to niggli_reduce as a row with three colum  matrices,
same as most DFT softwares' lattice inputs.

In the implementation details, since the lattice is represented by a row vector,
the transformation operation on the lattice is left-multiplied, such as:

.. code-block:: python

    import numpy as np

    # TMatrix is the transform matrix
    new_Lattice = np.matmul(TMatrix, old_Lattice)

For details of the algorithm, see [[Niggli for 2d and 3d]](http://)

Install
----------

.. code-block:: shell

    $ pip install pniggli

Usage
----------

.. code-block:: python

    from pniggli import niggli_reduce, niggli_check

    lattice_3D = [4.912, 0.000, 0.000,
                -2.456, 4.254, 0.000,
                0.000, 0.000, 0.000]
    niggli_lattice = niggli_reduce(lattice_3D)
    print(niggli_lattice)
    # Out:
    # array([[ 4.912,  0.   ,  0.   ],
    #        [-2.456,  4.254,  0.   ],
    #        [ 0.   ,  0.   , 16.   ]])
    print(niggli_check(niggli_lattice)) # True

    lattice_2D = [2.4560000896, 0.0000000000,
                11.0520002567, 2.1269502021]
    niggli_lattice = niggli_reduce(lattice_2D)
    print(niggli_lattice)
    # Out[6]:
    # array([[-1.2279999 , -2.1269502 ],
    #        [-1.22800019,  2.1269502 ]])


The 2D example is a triangle motif.

Version
----------

v0.1.2
########
+ 2D and 3D niggli reduce support
+ niggli_check for 3D lattice

v0.1.0
#######
+ 3D niggli reduce support
+ niggli_check for 3D lattice
