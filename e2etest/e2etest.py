#!/usr/bin/env python
# coding: utf-8

__author__ = "Martha Brennich"
__license__ = "MIT"
__copyright__ = "2020"
__date__ = "11/07/2020"

import unittest
import e2etest_freesas



def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(e2etest_freesas.suite())
    return testSuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
