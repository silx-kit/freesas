#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2017-2024  European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "02/10/2026"

import logging
import time
import unittest

import numpy

from .._bift import (
    BIFT,
    distribution_parabola,
    distribution_sphere,
    ensure_edges_zero,
    smooth_density,
)
from ..bift import auto_bift

try:
    from numpy import trapezoid  # numpy 2
except ImportError:
    from numpy import trapz as trapezoid  # numpy1

logger = logging.getLogger(__name__)


class TestBIFT(unittest.TestCase):
    DMAX = 10
    NPT = 100
    SIZE = 1000

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = numpy.linspace(0, cls.DMAX, cls.NPT + 1)
        dr = cls.DMAX / cls.NPT
        cls.p = -cls.r * (cls.r - cls.DMAX)  # Nice parabola
        q = numpy.linspace(0, 8 * cls.DMAX / 3, cls.SIZE + 1)
        sincqr = numpy.sinc(numpy.outer(q, cls.r / numpy.pi))
        intensity = 4 * numpy.pi * (cls.p * sincqr).sum(axis=-1) * dr
        err = numpy.sqrt(intensity)
        cls.I0 = intensity[0]
        cls.q = q[1:]
        cls.I = intensity[1:]
        cls.err = err[1:]
        cls.Rg = numpy.sqrt(
            0.5 * trapezoid(cls.p * cls.r**2, cls.r) / trapezoid(cls.p, cls.r)
        )
        print(cls.Rg)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.r = cls.p = cls.I = cls.q = cls.err = None

    def test_autobift(self):
        data = numpy.vstack((self.q, self.I, self.err)).T
        t0 = time.perf_counter()
        bo = auto_bift(data)
        key, _value, _valid = bo.get_best()
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

    def test_background(self):
        """A flat background added to the data must be recovered by fit_background."""
        bg = 0.05 * numpy.median(numpy.abs(self.I[self.q > 0.7 * self.q.max()]))
        npt = 64
        ref = numpy.vstack((self.q, self.I, self.err)).T
        data = numpy.vstack((self.q, self.I + bg, self.err)).T

        # without the option, the background is simply not fitted
        stats = auto_bift(data, npt=npt, fit_background=False).calc_stats()
        self.assertIsNone(stats.background_avg, "no background reported by default")

        # The absolute level also soaks up the model error (p(r) sampled on npt points),
        # so the check is differential: the shift of the fitted background has to match
        # the level which was added to the data.
        b_ref = auto_bift(ref, npt=npt, fit_background=True).calc_stats().background_avg
        stats = auto_bift(data, npt=npt, fit_background=True).calc_stats()
        self.assertIsNotNone(stats.background_avg, "background is reported")
        self.assertAlmostEqual(
            (stats.background_avg - b_ref) / bg, 1, 3, "flat background is recovered"
        )

    def test_disributions(self):
        pp = numpy.asarray(distribution_parabola(self.I0, self.DMAX, self.NPT))
        ps = numpy.asarray(distribution_sphere(self.I0, self.DMAX, self.NPT))
        self.assertAlmostEqual(
            trapezoid(ps, self.r) * 4 * numpy.pi / self.I0,
            1,
            3,
            "Distribution for a sphere looks OK",
        )
        self.assertAlmostEqual(
            trapezoid(pp, self.r) * 4 * numpy.pi / self.I0,
            1,
            3,
            "Distribution for a parabola looks OK",
        )
        self.assertTrue(numpy.allclose(pp, self.p, 1e-4), "distribution matches")

    def test_fixEdges(self):
        ones = numpy.ones(self.NPT)
        ensure_edges_zero(ones)
        self.assertAlmostEqual(ones[0], 0, msg="1st point set to 0")
        self.assertAlmostEqual(ones[-1], 0, msg="last point set to 0")
        self.assertTrue(
            numpy.allclose(ones[1:-1], numpy.ones(self.NPT - 2), 1e-7),
            msg="non-edge points unchanged",
        )

    def test_smoothing(self):
        ones = numpy.ones(self.NPT)
        empty = numpy.empty(self.NPT)
        smooth_density(ones, empty)
        self.assertTrue(
            numpy.allclose(ones, empty, 1e-7), msg="flat array smoothed into flat array"
        )
        random = numpy.random.rand(self.NPT)
        smooth = numpy.empty(self.NPT)
        smooth_density(random, smooth)
        self.assertAlmostEqual(
            random[0],
            smooth[0],
            msg="first points of random array and smoothed random array match",
        )
        self.assertAlmostEqual(
            random[-1],
            smooth[-1],
            msg="last points of random array and smoothed random array match",
        )
        self.assertTrue(
            smooth[1] >= min(smooth[0], smooth[2])
            and smooth[1] <= max(smooth[0], smooth[2]),
            msg="second point of random smoothed array between 1st and 3rd",
        )
        self.assertTrue(
            smooth[-2] >= min(smooth[-1], smooth[-3])
            and smooth[-2] <= max(smooth[-1], smooth[-3]),
            msg="second to last point of random smoothed array between 3rd to last and last",
        )
        sign = numpy.sign(random[1:-3] - smooth[2:-2]) * numpy.sign(
            smooth[2:-2] - random[3:-1]
        )
        self.assertTrue(
            numpy.allclose(sign, numpy.ones(self.NPT - 4), 1e-7),
            msg="central points of random array and smoothed random array alternate",
        )


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestBIFT("test_disributions"))
    testSuite.addTest(TestBIFT("test_autobift"))
    testSuite.addTest(TestBIFT("test_background"))
    testSuite.addTest(TestBIFT("test_fixEdges"))
    testSuite.addTest(TestBIFT("test_smoothing"))
    return testSuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
