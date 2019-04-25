# Test Data transformed from https://github.com/atztogo/niggli/ repository
# However use row vector to represent the lattice
import unittest
import os
import numpy as np
from pniggli import niggli_reduce, niggli_check

LATTICE_FILENAME_3D = os.path.join(os.path.dirname(__file__), 'lattices_3d.dat')
RLATTICE_FILENAME_3D = os.path.join(os.path.dirname(__file__), 'reduced_lattices_3d.dat')

LATTICE_FILENAME_2D = os.path.join(os.path.dirname(__file__), 'lattices_2d.dat')
RLATTICE_FILENAME_2D = os.path.join(os.path.dirname(__file__), 'reduced_lattices_2d.dat')

def get_lattice_parameters(lattice):
    return np.sqrt(np.matmul(lattice, lattice.T).diagonal())


def read_file(filename):
    all_lattices = []
    with open(filename) as f:
        lattice = []
        for line in f:
            if line[0] == '#':
                continue
            row = [float(x) for x in line.split()]
            lattice.append(row)
            if len(lattice) == len(row):
                all_lattices.append(np.array(lattice))
                lattice = []
    return all_lattices

def show_lattice(i, lattice):
    print("# %d" % (i + 1))
    for v in lattice:
        print(" ".join(["%20.16f" % x for x in v]))

class TestNiggli3D(unittest.TestCase):

    def setUp(self):
        super(TestNiggli3D, self).setUp()
        self._input_lattices = read_file(LATTICE_FILENAME_3D)
        self._reference_lattices = read_file(RLATTICE_FILENAME_3D)

    def tearDown(self):
        pass

    def test_lattice_error(self):
        lattice = [1, 2.0, 2.0]
        with self.assertRaises(ValueError):
            niggli_reduce(lattice)

    def test_reference_data(self):
        for i, reference_lattice in enumerate(self._reference_lattices):
            angles = np.array(self.get_angles(reference_lattice))
            self.assertTrue((angles > 90 - 1e-3).all() or
                            (angles < 90 + 1e-3).all(),
                            msg=("%d %s" % (i + 1, angles)))

    def test_niggli_check_and_data(self):
        for i, reference_lattice in enumerate(self._reference_lattices):
            is_niggli = niggli_check(reference_lattice, 1e-5)
            self.assertTrue(is_niggli, msg=("%d not niggli" % (i + 1)))

    def test_niggli_reduce(self):
        for i, (input_lattice, reference_lattice) in enumerate(
                zip(self._input_lattices, self._reference_lattices)):
            reduced_lattice = niggli_reduce(input_lattice)
            show_lattice(i, reduced_lattice)
            self.assertTrue(
                np.allclose(reduced_lattice, reference_lattice),
                msg="\n".join(
                    ["# %d" % (i + 1),
                     "Input lattice",
                     "%s" % input_lattice,
                     " angles: %s" % np.array(self.get_angles(input_lattice)),
                     "Reduced lattice in reference data",
                     "%s" % reference_lattice,
                     " angles: %s" % np.array(self.get_angles(reference_lattice)),
                     "Reduced lattice",
                     "%s" % reduced_lattice,
                     " angles: %s" % np.array(self.get_angles(reduced_lattice))]))
    def get_angles(self, lattice):
        a, b, c = get_lattice_parameters(lattice)
        alpha = np.arccos(np.dot(lattice[1,:], lattice[2,:]) / b / c)
        beta  = np.arccos(np.dot(lattice[2,:], lattice[0,:]) / c / a)
        gamma = np.arccos(np.dot(lattice[0,:], lattice[1,:]) / a / b)
        return np.array([alpha, beta, gamma]) / np.pi * 180


class TestNiggli2D(unittest.TestCase):

    def setUp(self):
        super(TestNiggli2D, self).setUp()
        self._input_lattices = read_file(LATTICE_FILENAME_2D)
        self._reference_lattices = read_file(RLATTICE_FILENAME_2D)

    def tearDown(self):
        pass

    def test_lattice_error(self):
        lattice = [1, 2.0, 2.0, 2.3, 0]
        with self.assertRaises(ValueError):
            niggli_reduce(lattice)

    def test_niggli_reduce(self):
        for i, (input_lattice, reference_lattice) in enumerate(
                zip(self._input_lattices, self._reference_lattices)):
            reduced_lattice = niggli_reduce(input_lattice)
            show_lattice(i, reduced_lattice)
            self.assertTrue(
                np.allclose(reduced_lattice, reference_lattice),
                msg="\n".join(
                    ["# %d" % (i + 1),
                     "Input lattice",
                     "%s" % input_lattice,
                     " angles: %s" % np.array(self.get_angles(input_lattice)),
                     "Reduced lattice in reference data",
                     "%s" % reference_lattice,
                     " angles: %s" % np.array(self.get_angles(reference_lattice)),
                     "Reduced lattice",
                     "%s" % reduced_lattice,
                     " angles: %s" % np.array(self.get_angles(reduced_lattice))]))

    def get_angles(self, lattice):
        a, b = get_lattice_parameters(lattice)
        theta = np.arccos(np.dot(lattice[0,:], lattice[1,:]) / a / b)
        return theta / np.pi * 180
