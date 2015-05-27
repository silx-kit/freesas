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
    testfile = join(base, "testdata", "model-01.pdb")

    def test_fineness(self):
        m = SASModel()
        m.read(self.testfile)
        f_np = m._calc_fineness(False)
        f_cy = m._calc_fineness(True)
        self.assertEqual(f_np,f_cy,"fineness is the same %s!=%s"%(f_np, f_cy))

def test_suite_all_distance():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestDistance("test_fineness"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_distance()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)