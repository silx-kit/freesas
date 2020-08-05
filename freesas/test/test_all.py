#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"
__date__ = "25/04/2020"

import unittest
from . import test_model
from . import test_align
from . import test_distance
from . import test_cormap
from . import test_autorg
from . import test_bift
from . import test_sasio


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_bift.suite())
    testSuite.addTest(test_model.suite())
    testSuite.addTest(test_align.suite())
    testSuite.addTest(test_distance.suite())
    testSuite.addTest(test_cormap.suite())
    testSuite.addTest(test_autorg.suite())
    testSuite.addTest(test_sasio.suite())
    return testSuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
