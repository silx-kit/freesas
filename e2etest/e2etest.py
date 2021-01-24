#!/usr/bin/env python
# coding: utf-8

"""Run the end to end tests of the project."""

__author__ = "Martha Brennich"
__license__ = "MIT"
__copyright__ = "2020"
__date__ = "11/07/2020"

import sys
import unittest
import e2etest_freesas, e2etest_guinier_apps


def suite():
    """Creates suite for e2e tests"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(e2etest_freesas.suite())
    test_suite.addTest(e2etest_guinier_apps.suite())
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    result = runner.run(suite())

    if result.wasSuccessful():
        EXIT_STATUS = 0
    else:
        EXIT_STATUS = 1

    sys.exit(EXIT_STATUS)
