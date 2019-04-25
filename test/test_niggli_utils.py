import unittest

import numpy as np

from pniggli.utils import _get_G_param, _get_metric, _get_angle_type, _get_angle_param

class TestNiggliUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_angle_type(self):
        angle = 0.0
        t = _get_angle_type(angle, 1e-5)
        self.assertEqual(t, 0)

        angle = 2.0
        t = _get_angle_type(angle, 1e-5)
        self.assertEqual(t, 1)

        angle = -1.02
        t = _get_angle_type(angle, 1e-5)
        self.assertEqual(t, -1)

    def test_get_angle_param(self):
        lattice = np.array([4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0]).reshape((3, 3))
        G = _get_metric(lattice)
        _, _, _, X, E, Z = _get_G_param(G)
        l, m, n = _get_angle_param(X, E, Z, eps=1e-5)

        self.assertEqual(l, 0)
        self.assertEqual(m, 0)
        self.assertEqual(n, 0)

    def test_get_param(self):
        pass

    def test_get_metric(self):
        pass
