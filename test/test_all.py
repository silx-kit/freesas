__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import unittest
from test_model import test_suite_all_model
from test_align import test_suite_all_alignment
from test_distance import test_suite_all_distance


def test_suite_all():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_suite_all_model())
    testSuite.addTest(test_suite_all_alignment())
    testSuite.addTest(test_suite_all_distance())
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
