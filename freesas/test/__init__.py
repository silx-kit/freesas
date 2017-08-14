#!usr/bin/env python
# coding: utf-8

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "16/12/2015"
__copyright__ = "2015, ESRF"

import unittest
from .test_all import suite


def run():
    runner = unittest.TextTestRunner()
    return runner.run(suite())


if __name__ == '__main__':
    run()