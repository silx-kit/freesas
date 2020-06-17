# -*- coding: utf-8 -*-
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2017  European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__authors__ = ["J. Kieffer"]
__license__ = "MIT"
__date__ = "10/06/2020"

import numpy
import unittest
from .utilstests import get_datafile
from ..autorg import autoRg, RG_RESULT, linear_fit, auto_gpa, auto_guinier
from ..invariants import calc_Rambo_Tainer
from .._bift import distribution_sphere
from math import sqrt, pi
from scipy.stats import linregress
import logging
logger = logging.getLogger(__name__)


class TestAutoRg(unittest.TestCase):
    testfile = get_datafile("bsa_005_sub.dat")
    # Reference implementation
    atsas_autorg = {"Version": "Atsas 2.6.1",
                    "Rg": 2.98016,
                    "sigma_Rg": 0.156859,
                    "I0": 61.3093,
                    "sigma_I0": 0.0606315,
                    "start_point": 46,
                    "end_point": 95,
                    "quality": 0.752564,
                    "aggregated": 0}

    def test_atsas(self):
        logger.info("test file: %s", self.testfile)
        data = numpy.loadtxt(self.testfile)
        atsas_result = self.atsas_autorg.copy()
        logger.debug("Reference version: %s" % atsas_result.pop("Version"))
        atsas_result = RG_RESULT(**atsas_result)
        free_result = autoRg(data)
        logger.debug("Ref: %s" % (atsas_result,))
        logger.debug("Obt: %s" % (free_result,))
        self.assertAlmostEqual(atsas_result.Rg, free_result.Rg, 1, "RG fits within 2 digits")
        self.assertAlmostEqual(atsas_result.I0, free_result.I0, msg="I0 fits within +/- 1 ", delta=1)
        self.assertAlmostEqual(atsas_result.quality, free_result.quality, 0, msg="quality fits within 0 digits")

    def test_synthetic(self):
        "Test based on sythetic data: a sphere of radius R0=4 which Rg should be 4*sqrt(3/5)"
        R0 = 4
        npt = 1000
        I0 = 1e2
        Dmax = 2 * R0
        size = 5000
        r = numpy.linspace(0, Dmax, npt + 1)
        p = distribution_sphere(I0, Dmax, npt)
        q = numpy.linspace(0, 10, size)
        qr = numpy.outer(q, r / pi)
        T = (4 * pi * (r[-1] - r[0]) / npt) * numpy.sinc(qr)
        I = T.dot(p)
        err = numpy.sqrt(I)
        data = numpy.vstack((q, I, err)).T
        Rg = autoRg(data)
        logger.info("auto_rg %s", Rg)
        self.assertAlmostEqual(R0 * sqrt(3 / 5), Rg.Rg, 0, "Rg matches for a sphere")
        self.assertGreater(R0 * sqrt(3 / 5), Rg.Rg - Rg.sigma_Rg, "Rg in range matches for a sphere")
        self.assertLess(R0 * sqrt(3 / 5), Rg.Rg + Rg.sigma_Rg, "Rg in range matches for a sphere")
        self.assertAlmostEqual(I0, Rg.I0, 0, "I0 matches for a sphere")
        self.assertGreater(I0, Rg.I0 - Rg.sigma_I0, "I0 matches for a sphere")
        self.assertLess(I0, Rg.I0 + Rg.sigma_I0, "I0 matches for a sphere")

        gpa = auto_gpa(data)
        logger.info("auto_gpa %s", gpa)
        self.assertAlmostEqual(gpa.Rg / (R0 * sqrt(3. / 5)), 1.00, 0, "Rg matches for a sphere")
        self.assertAlmostEqual(gpa.I0 / I0, 1.0, 1, "I0 matches for a sphere")

        guinier = auto_guinier(data)
        logger.info("auto_guinier %s", guinier)
        self.assertAlmostEqual(R0 * sqrt(3. / 5), guinier.Rg, 0, "Rg matches for a sphere")
        sigma_Rg = max(guinier.sigma_Rg, 1e-4)
        sigma_I0 = max(guinier.sigma_I0, 1e-4)
        self.assertGreater(R0 * sqrt(3. / 5), guinier.Rg - sigma_Rg, "Rg in range matches for a sphere")
        self.assertLess(R0 * sqrt(3. / 5), guinier.Rg + sigma_Rg, "Rg in range matches for a sphere")
        self.assertAlmostEqual(I0, guinier.I0, 0, "I0 matches for a sphere")
        self.assertGreater(I0, guinier.I0 - sigma_I0, "I0 matches for a sphere")
        self.assertLess(I0, guinier.I0 + sigma_I0, "I0 matches for a sphere")

        # Check RT invarients...
        rt = calc_Rambo_Tainer(data, guinier)
        self.assertIsNotNone(rt, "Rambo-Tainer invariants are actually calculated")


class TestFit(unittest.TestCase):
    # Testcase originally comes from wikipedia article on linear regression, expected results from scipy.stats.linregress

    def test_linear_fit_static(self):
        testx = [1.47, 1.5, 1.52, 1.55, 1.57, 1.6, 1.63, 1.65, 1.68, 1.7, 1.73, 1.75, 1.78, 1.80, 1.83]
        testy = [52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.1, 69.92, 72.19, 74.46]
        testw = [1.0] * 15
        testintercept = -39.061956
        testslope = +61.2721865
        fit_result = linear_fit(testx, testy, testw)
        # print(fit_result)
        self.assertAlmostEqual(fit_result.intercept, testintercept, 5, "Intercept fits wihtin 4(?) digits")
        self.assertAlmostEqual(fit_result.slope, testslope, 5, "Intercept fits wihtin 4(?) digits")

    def test_linspace(self):
        size = 100
        x = numpy.linspace(-10, 10, size)
        y = numpy.linspace(10, 0, size)
        w = numpy.random.random(size)
        fit_result = linear_fit(x, y, w)
        # print(fit_result)
        self.assertAlmostEqual(fit_result.intercept, 5, 5, "Intercept fits wihtin 4(?) digits")
        self.assertAlmostEqual(fit_result.slope, -0.5, 5, "Intercept fits wihtin 4(?) digits")

    def test_random(self):
        size = 100
        x = numpy.random.random(size)
        y = 1.6 * x + 5 + numpy.random.random(size)
        w = numpy.ones(size)
        fit_result = linear_fit(x, y, w)
        ref = linregress(x, y)
        self.assertAlmostEqual(fit_result.intercept, ref[1], 5, "Intercept fits wihtin 4(?) digits")
        self.assertAlmostEqual(fit_result.slope, ref[0], 5, "Intercept fits wihtin 4(?) digits")
        self.assertAlmostEqual(fit_result.R2, ref.rvalue ** 2, 5, "R² value matcheswihtin 4(?) digits")


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAutoRg("test_atsas"))
    testSuite.addTest(TestAutoRg("test_synthetic"))
    testSuite.addTest(TestFit("test_linear_fit_static"))
    testSuite.addTest(TestFit("test_linspace"))
    testSuite.addTest(TestFit("test_linear_fit_static"))
    return testSuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
