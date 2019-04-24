# TestCase from https://github.com/atztogo/niggli
import unittest
import os
import numpy as np
from pniggli import niggli_reduce

LATTICE_FILENAME = os.path.join(os.path.dirname(__file__), 'lattices.dat')
RLATTICE_FILENAME = os.path.join(os.path.dirname(__file__), 'reduced_lattices.dat')

class TestNiggli(unittest.TestCase):

    def setUp(self):
        super(TestNiggli, self).setUp()
        self._input_lattices = read_file(LATTICE_FILENAME)
        self._reference_lattices = read_file(RLATTICE_FILENAME)

    def tearDown(self):
        pass

    def test_reference_data(self):
        for i, reference_lattice in enumerate(self._reference_lattices):
            angles = np.array(get_angles(reference_lattice))
            self.assertTrue((angles > 90 - 1e-3).all() or
                            (angles < 90 + 1e-3).all(),
                            msg=("%d %s" % (i + 1, angles)))

    def test_niggli_reduce(self):
        for i, (input_lattice, reference_lattice) in enumerate(
                zip(self._input_lattices, self._reference_lattices)):
            reduced_lattice = niggli_reduce(input_lattice)
            # self._show_lattice(i, reduced_lattice)
            self.assertTrue(
                np.allclose(reduced_lattice, reference_lattice),
                msg="\n".join(
                    ["# %d" % (i + 1),
                     "Input lattice",
                     "%s" % input_lattice,
                     " angles: %s" % np.array(get_angles(input_lattice)),
                     "Reduced lattice in reference data",
                     "%s" % reference_lattice,
                     " angles: %s" % np.array(get_angles(reference_lattice)),
                     "Reduced lattice",
                     "%s" % reduced_lattice,
                     " angles: %s" % np.array(get_angles(reduced_lattice))]))

def get_lattice_parameters(lattice):
    return np.sqrt(np.dot(lattice.T, lattice).diagonal())

def get_angles(lattice):
    a, b, c = get_lattice_parameters(lattice)
    alpha = np.arccos(np.vdot(lattice[:,1], lattice[:,2]) / b / c)
    beta  = np.arccos(np.vdot(lattice[:,2], lattice[:,0]) / c / a)
    gamma = np.arccos(np.vdot(lattice[:,0], lattice[:,1]) / a / b)
    return np.array([alpha, beta, gamma]) / np.pi * 180

def read_file(filename):
    all_lattices = []
    with open(filename) as f:
        lattice = []
        for line in f:
            if line[0] == '#':
                continue
            lattice.append([float(x) for x in line.split()])
            if len(lattice) == 3:
                all_lattices.append(np.transpose(lattice))
                lattice = []
    return all_lattices

def show_lattice(i, lattice):
    print("# %d" % (i + 1))
    for v in lattice.T:
        print(" ".join(["%20.16f" % x for x in v]))