#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"
__date__ = "05/09/2017"

import unittest
from . import test_model
from . import test_align
from . import test_distance
from . import test_cormap
from . import test_autorg


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_model.suite())
    testSuite.addTest(test_align.suite())
    testSuite.addTest(test_distance.suite())
    testSuite.addTest(test_cormap.suite())
    testSuite.addTest(test_autorg.suite())
    return testSuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
