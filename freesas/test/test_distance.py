#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2015, ESRF"
__date__ = "16/12/2015"

import numpy
import unittest
from .utilstests import get_datafile
from ..model import SASModel
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cdistance_test")


class TestDistance(unittest.TestCase):
    testfile1 = get_datafile("model-01.pdb")
    testfile2 = get_datafile("dammif-01.pdb")

    def test_invariants(self):
        m = SASModel()
        m.read(self.testfile1)
        f_np, r_np, d_np = m.calc_invariants(False)
        f_cy, r_cy, d_cy = m.calc_invariants(True)
        self.assertAlmostEqual(f_np, f_cy, 10, "fineness is the same %s!=%s" % (f_np, f_cy))
        self.assertAlmostEqual(r_np, r_cy, 10, "Rg is the same %s!=%s" % (r_np, r_cy))
        self.assertAlmostEqual(d_np, d_cy, 10, "Dmax is the same %s!=%s" % (d_np, d_cy))

    def test_distance(self):
        m = SASModel()
        n = SASModel()
        m.read(self.testfile1)
        n.read(self.testfile2)
        f_np = m.dist(n, m.atoms, n.atoms, False)
        f_cy = m.dist(n, m.atoms, n.atoms, True)
        self.assertAlmostEqual(f_np, f_cy, 10, "distance is the same %s!=%s" % (f_np, f_cy))

    def test_same(self):
        m = SASModel()
        n = SASModel()
        m.read(self.testfile1)
        n.read(self.testfile1)
        numpy.random.shuffle(n.atoms)
        f_np = m.dist(n, m.atoms, n.atoms, False)
        f_cy = m.dist(n, m.atoms, n.atoms, True)
        self.assertAlmostEqual(f_np, 0, 10, "NSD not nul with np")
        self.assertAlmostEqual(f_cy, 0, 10, "NSD not nul with cy")


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestDistance("test_invariants"))
    testSuite.addTest(TestDistance("test_distance"))
    testSuite.addTest(TestDistance("test_same"))
    return testSuite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
