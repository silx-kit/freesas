# -*- coding: utf-8 -*-
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2017  European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "25/07/2020"

import unittest
import logging
from numpy import array, allclose
from ..sasio import parse_ascii_data #,load_scattering_data
logger = logging.getLogger(__name__)


class TestSasIO(unittest.TestCase):

    def test_parse_3_ok(self):
        """
        Test for successful parsing of file with some invalid lines
        """
        file_content = ["Test data for",
                        "file parsing",
                        "1 1 1",
                        "2 a 2",
                        "3 3 3",
                        "some stuff at the end",
                        ]
        expected_result = array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        data = parse_ascii_data(file_content, number_of_columns=3)
        self.assertTrue(allclose(data, expected_result, 1e-7),
                        msg="3 column parse returns expected result")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestSasIO("test_parse_3_ok"))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
