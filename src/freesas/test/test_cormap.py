#!/usr/bin/python
# coding: utf-8
from __future__ import print_function

__author__ = "Jerome"
__license__ = "MIT"
__copyright__ = "2017, ESRF"

import numpy
import unittest

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_cormap")
from .. import cormap


class TestCormap(unittest.TestCase):

    def test_longest(self):
        size = 1000
        target = 50
        start = 100

        data = numpy.ones(size, dtype="float32")
        res = cormap.measure_longest(data)
        self.assertEqual(res, size, msg="computed size is correct: positive")

        data -= 2
        res = cormap.measure_longest(data)
        self.assertEqual(res, size, msg="computed size is correct: negative")

        data[:] = 0
        data[start: start + target] = 1.0
        res = cormap.measure_longest(data)
        self.assertEqual(res, target, msg="computed size is correct: positive/zero")
        data = numpy.zeros(size, dtype="float32")
        data[start: start + target] = -1.0
        res = cormap.measure_longest(data)
        self.assertEqual(res, target, msg="computed size is correct: negative/zero")
        data = numpy.fromfunction(lambda n:(-1) ** n, (size,))
        data[start: start + target] = 1.0
        res = cormap.measure_longest(data)
        self.assertEqual(res, target + 1, msg="computed size is correct: positive/alternating")
        data = numpy.fromfunction(lambda n:(-1) ** n, (size,))
        data[start: start + target] = -1.0
        res = cormap.measure_longest(data)
        self.assertEqual(res, target + 1, msg="computed size is correct: negative/alternating")

    def test_stats(self):
        self.assertEqual(cormap.LROH.A(10, 0), 1)
        self.assertEqual(cormap.LROH.A(10, 1), 144)
        self.assertEqual(cormap.LROH.A(10, 2), 504)
        self.assertEqual(cormap.LROH.A(10, 10), 1024)
        self.assertEqual(cormap.LROH.A(10, 11), 1024)

        self.assertEqual(cormap.LROH.A(0, 3), 1)
        self.assertEqual(cormap.LROH.A(1, 3), 2)
        self.assertEqual(cormap.LROH.A(2, 3), 4)
        self.assertEqual(cormap.LROH.A(3, 3), 8)
        self.assertEqual(cormap.LROH.A(4, 3), 15)
        self.assertEqual(cormap.LROH.A(5, 3), 29)
        self.assertEqual(cormap.LROH.A(6, 3), 56)
        self.assertEqual(cormap.LROH.A(7, 3), 108)
        self.assertEqual(cormap.LROH.A(8, 3), 208)

        self.assertAlmostEqual(cormap.LROH(200, 0), 1)
        self.assertAlmostEqual(cormap.LROH(200, 4), 0.97, 2)
        self.assertAlmostEqual(cormap.LROH(200, 5), 0.80, 2)
        self.assertAlmostEqual(cormap.LROH(200, 6), 0.54, 2)
        self.assertAlmostEqual(cormap.LROH(200, 7), 0.32, 2)
        self.assertAlmostEqual(cormap.LROH(200, 8), 0.17, 2)
        self.assertAlmostEqual(cormap.LROH(200, 9), 0.09, 2)
        self.assertAlmostEqual(cormap.LROH(200, 10), 0.05, 2)
        self.assertAlmostEqual(cormap.LROH(200, 11), 0.02, 2)


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestCormap("test_longest"))
    testSuite.addTest(TestCormap("test_longest"))
    testSuite.addTest(TestCormap("test_stats"))
    return testSuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
