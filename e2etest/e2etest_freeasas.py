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

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "11/07/2020"

import unittest

import logging
logger = logging.getLogger(__name__)
import time
import re


expectedTexts = {"Label of scatter plot X-axis": ["$q$ (nm$^{-1}$)"],
                 "Label of scatter plot Y-axis": ["$I(q)$ (log scale)"],
                 "Smallest Tick of scatter plot Y axis": ["$\mathdefault{10^{-2}}$"],
                 "Largest Tick of scatter plot Y axis": ["$\mathdefault{10^{2}}$"],
                 "Scattering plot caption": ["Scattering curve"],
                 "Experimental data legend": ["Experimental data"], #used twice, but we might just ignore that
                 "Guinier region legend": ["Guinier region: $R_g=$",
                                           "[23].[0-9][0-9]nm", "6[0-9].[0-9][0-9]"],
                 "BIFT fit legend": ["BIFT extraplolated: $D_{max}=$",
                                     "9.[0-9][0-9] nm"],
                 "Label of Guinier plot X-axis": ["$q^2$ (nm$^{-2}$)"],
                 "Label of Guinier plot Y-axis": ["ln[$I(q)$]"],
                 "Guinier - qRgmin": ["$(qR_{g})_{min}$=",
                                      "0.9"],
                 "Guinier - qRgmax": ["$(qR_{g})_{max}$=",
                                      "1.3"],
                 "Guinier region label": ["Guinier region"],
                 "Guinier plot caption": ["Guinier plot: $R_{g}=$" ,
                                          "3.00 nm $I_{0}=$61.72"],
                 "Guinier fit equation": ["ln[$I(q)$] = 4.12 -3.01 * $q^2$"],
                 "Guinier fit data label": ["Experimental curve"],
                 "Label of Kratky plot X-Axis":["$qR_{g}$"],
                 "Label of Kratky plot Y-Axis":["$(qR_{g})^2 I/I_{0}$"],
                 "Kratky plot caption": ["Dimensionless Kratky plot"],
                 "Label of distribution plot X-axis": ["$r$ (nm)"],
                 }


class TestBIFT(unittest.TestCase):

    DMAX = 10
    NPT = 100
    SIZE = 1000

    @classmethod
    def setUpClass(cls):
        super(TestBIFT, cls).setUpClass()
        cls.r = numpy.linspace(0, cls.DMAX, cls.NPT + 1)
        dr = cls.DMAX / cls.NPT
        cls.p = -cls.r * (cls.r - cls.DMAX)  # Nice parabola
        q = numpy.linspace(0, 8 * cls.DMAX / 3, cls.SIZE + 1)
        sincqr = numpy.sinc(numpy.outer(q, cls.r / numpy.pi))
        I = 4 * numpy.pi * (cls.p * sincqr).sum(axis=-1) * dr
        err = numpy.sqrt(I)
        cls.I0 = I[0]
        cls.q = q[1:]
        cls.I = I[1:]
        cls.err = err[1:]
        cls.Rg = numpy.sqrt(0.5 * numpy.trapz(cls.p * cls.r ** 2, cls.r) / numpy.trapz(cls.p, cls.r))
        print(cls.Rg)

    @classmethod
    def tearDownClass(cls):
        super(TestBIFT, cls).tearDownClass()
        cls.r = cls.p = cls.I = cls.q = cls.err = None

    def test_autobift(self):
        data = numpy.vstack((self.q, self.I, self.err)).T
        t0 = time.perf_counter()
        bo = auto_bift(data)
        key, value, valid = bo.get_best()
#         print("key is ", key)
        stats = bo.calc_stats()
#         print("stat is ", stats)
        logger.info("Auto_bift time: %s", time.perf_counter() - t0)
        self.assertAlmostEqual(self.DMAX / key.Dmax, 1, 1, "DMax is correct")
        self.assertAlmostEqual(self.I0 / stats.I0_avg, 1, 1, "I0 is correct")
        self.assertAlmostEqual(self.Rg / stats.Rg_avg, 1, 2, "Rg is correct")

    def test_BIFT(self):
        bift = BIFT(self.q, self.I, self.err)
        # test two scan functions
        key = bift.grid_scan(9, 11, 5, 10, 100, 5, 100)
        # print("key is ", key)
        self.assertAlmostEqual(self.DMAX / key.Dmax, 1, 2, "DMax is correct")
        res = bift.monte_carlo_sampling(10, 3, 100)
        # print("res is ", res)
        self.assertAlmostEqual(self.DMAX / res.Dmax_avg, 1, 4, "DMax is correct")

    def test_disributions(self):
        pp = numpy.asarray(distribution_parabola(self.I0, self.DMAX, self.NPT))
        ps = numpy.asarray(distribution_sphere(self.I0, self.DMAX, self.NPT))
        self.assertAlmostEqual(numpy.trapz(ps, self.r) * 4 * numpy.pi / self.I0, 1, 3, "Distribution for a sphere looks OK")
        self.assertAlmostEqual(numpy.trapz(pp, self.r) * 4 * numpy.pi / self.I0, 1, 3, "Distribution for a parabola looks OK")
        self.assertTrue(numpy.allclose(pp, self.p, 1e-4), "distribution matches")

    def test_fixEdges(self):
        ones = numpy.ones(self.NPT)
        ensure_edges_zero(ones)
        self.assertAlmostEqual(ones[0], 0,  msg="1st point set to 0")
        self.assertAlmostEqual(ones[-1], 0,  msg="last point set to 0")
        self.assertTrue(numpy.allclose(ones[1:-1], numpy.ones(self.NPT-2), 1e-7), msg="non-edge points unchanged")

    def test_smoothing(self):
        ones = numpy.ones(self.NPT)
        empty = numpy.empty(self.NPT)
        smooth_density(ones,empty)
        self.assertTrue(numpy.allclose(ones, empty, 1e-7), msg="flat array smoothed into flat array")
        random = numpy.random.rand(self.NPT)
        smooth =  numpy.empty(self.NPT)
        smooth_density(random,smooth)
        self.assertAlmostEqual(random[0], smooth[0],  msg="first points of random array and smoothed random array match")
        self.assertAlmostEqual(random[-1], smooth[-1],  msg="last points of random array and smoothed random array match")
        self.assertTrue(smooth[1]>=min(smooth[0], smooth[2]) and smooth[1]<=max(smooth[0], smooth[2]), msg="second point of random smoothed array between 1st and 3rd")
        self.assertTrue(smooth[-2]>=min(smooth[-1], smooth[-3]) and smooth[-2]<= max(smooth[-1], smooth[-3]), msg="second to last point of random smoothed array between 3rd to last and last")
        sign = numpy.sign(random[1:-3] - smooth[2:-2]) * numpy.sign(smooth[2:-2] - random[3:-1])
        self.assertTrue(numpy.allclose(sign, numpy.ones(self.NPT-4), 1e-7), msg="central points of random array and smoothed random array alternate")

def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestBIFT("test_disributions"))
    testSuite.addTest(TestBIFT("test_autobift"))
    testSuite.addTest(TestBIFT("test_fixEdges"))
    testSuite.addTest(TestBIFT("test_smoothing"))
    return testSuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
