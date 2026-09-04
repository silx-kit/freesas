__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"
__date__ = "03/07/2024"

import unittest

from . import (
    test_align,
    test_autorg,
    test_bift,
    test_cormap,
    test_distance,
    test_dnn,
    test_fitting,
    test_model,
    test_resources,
    test_sas_argparser,
    test_sasio,
)


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_bift.suite())
    testSuite.addTest(test_model.suite())
    testSuite.addTest(test_align.suite())
    testSuite.addTest(test_distance.suite())
    testSuite.addTest(test_cormap.suite())
    testSuite.addTest(test_autorg.suite())
    testSuite.addTest(test_sasio.suite())
    testSuite.addTest(test_sas_argparser.suite())
    testSuite.addTest(test_fitting.suite())
    testSuite.addTest(test_resources.suite())
    testSuite.addTest(test_dnn.suite())
    return testSuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
