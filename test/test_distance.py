#!/usr/bin/python
#coding: utf-8
__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import numpy
import unittest
import sys, os
from utilstests import base, join
from freesas.model import SASModel

class TestDistance(unittest.TestCase):
    testfile1 = join(base, "testdata", "model-01.pdb")
    testfile2 = join(base, "testdata", "dammif-01.pdb")

    def test_fineness(self):
        m = SASModel()
        m.read(self.testfile1)
        f_np = m._calc_fineness(False)
        f_cy = m._calc_fineness(True)
        self.assertEqual(f_np,f_cy,"fineness is the same %s!=%s"%(f_np, f_cy))

    def test_distance(self):
        m = SASModel()
        n = SASModel()
        m.read(self.testfile1)
        n.read(self.testfile2)
        f_np = m.dist(n, m.atoms, n.atoms, False)
        f_cy = m.dist(n, m.atoms, n.atoms, True)
        self.assertEqual(f_np,f_cy,"distance is the same %s!=%s"%(f_np, f_cy))

    def test_same(self):
        m = SASModel()
        n = SASModel()
        m.read(self.testfile1)
        n.read(self.testfile1)
        numpy.random.shuffle(n.atoms)
        f_np = m.dist(n, m.atoms, n.atoms, False)
        f_cy = m.dist(n, m.atoms, n.atoms, True)
        self.assertEqual(f_np, 0, "NSD not nul with np")
        self.assertEqual(f_cy, 0, "NSD not nul with cy")

def test_suite_all_distance():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestDistance("test_fineness"))
    testSuite.addTest(TestDistance("test_distance"))
    testSuite.addTest(TestDistance("test_same"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_distance()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)