#!usr/bin/env python
# coding: utf-8

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "15/01/2021"
__copyright__ = "2015-2021, ESRF"

import sys
import unittest
from .test_all import suite


def run_tests():
    """Run test complete test_suite"""
    mysuite = suite()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        print("Test suite failed")
        return 1
    else:
        print("Test suite succeeded")
        return 0


run = run_tests

if __name__ == '__main__':
    sys.exit(run_tests())
