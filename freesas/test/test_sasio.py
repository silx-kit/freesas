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
from sys import version_info
from unittest.mock import patch, mock_open
from numpy import array, allclose
from ..sasio import parse_ascii_data, load_scattering_data
logger = logging.getLogger(__name__)

#mock_open under 3.6 needs some patching up
# Code from https://github.com/python/cpython/blob/3.8/Lib/unittest/mock.py
if version_info.minor > 7:
    my_mock_open = mock_open
else:
    def my_mock_open(mock=None, read_data=''):
        import io
        _read_data = _to_stream(read_data)
        _state = [_read_data, None]
        basic_mock = mock_open(mock, read_data)
        handle = basic_mock.return_value

        def _to_stream(read_data):
            if isinstance(read_data, bytes):
                return io.BytesIO(read_data)
            else:
                return io.StringIO(read_data)

        def _iter_side_effect():
            if handle.readline.return_value is not None:
                while True:
                    yield handle.readline.return_value
            for line in _state[0]:
                yield line

        def _next_side_effect():
            if handle.readline.return_value is not None:
                return handle.readline.return_value
            return next(_state[0])

        handle.__iter__.side_effect = _iter_side_effect
        handle.__next__.side_effect = _next_side_effect

        return basic_mock


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

    def test_parse_no_data(self):
        """
        Test that an empty input list raises a ValueError
        """
        file_content = []
        with self.assertRaises(ValueError, msg="Empty list cannot be parsed"):
            parse_ascii_data(file_content, number_of_columns=3)

    def test_parse_no_valid_data(self):
        """
        Test that an input list with no valid data raises a ValueError
        """
        file_content = ["a a a", "2 4", "3 4 5 6", "# 3 4 6"]
        with self.assertRaises(ValueError,
                               msg="File with no float float float data"
                                   " cannot be parsed"):
            parse_ascii_data(file_content, number_of_columns=3)

    def test_load_clean_data(self):
        """
        Test that clean float float float data is loaded correctly.
        """
        file_content = ["# Test data for"
                        "# file parsing",
                        "1 1 1",
                        "2.0 2.0 1.0",
                        "3 3 3",
                        "#REMARK some stuff at the end",
                        ]
        expected_result = array([[1.0, 1.0, 1.0],
                                 [2.0, 2.0, 1.0],
                                 [3.0, 3.0, 3.0]])
        file_data = "\n".join(file_content)
        mocked_open = my_mock_open(read_data=file_data)
        with patch('builtins.open', mocked_open):
            with patch('numpy.DataSource.open', mocked_open):
                data = load_scattering_data("test")
        self.assertTrue(allclose(data, expected_result, 1e-7),
                        msg="Sunny data loaded correctly")

    def test_load_data_with_unescaped_header(self):
        """
        Test that an unescaped header does not hinder loading.
        """
        file_content = ["1 1 1",
                        "2.0 2.0 1.0",
                        "3 3 3",
                        "REMARK some stuff at the end"
                        ]
        expected_result = array([[1.0, 1.0, 1.0],
                                 [2.0, 2.0, 1.0],
                                 [3.0, 3.0, 3.0]])
        file_data = "\n".join(file_content)
        mocked_open = my_mock_open(read_data=file_data)
        with patch('builtins.open', mocked_open):
            with patch('numpy.DataSource.open', mocked_open):
                data = load_scattering_data("test")
        self.assertTrue(allclose(data, expected_result, 1e-7),
                        msg="Sunny data loaded correctly")

    def test_load_data_with_unescaped_footer(self):
        """
        Test that an unescaped footer does not hinder loading.
        """
        file_content = ["Test data for"
                        "file parsing",
                        "1 1 1",
                        "2.0 2.0 1.0",
                        "3 3 3",
                        ]
        expected_result = array([[1.0, 1.0, 1.0],
                                 [2.0, 2.0, 1.0],
                                 [3.0, 3.0, 3.0]])
        file_data = "\n".join(file_content)
        mocked_open = my_mock_open(read_data=file_data)
        with patch('builtins.open', mocked_open):
            with patch('numpy.DataSource.open', mocked_open):
                data = load_scattering_data("test")
        self.assertTrue(allclose(data, expected_result, 1e-7),
                        msg="Sunny data loaded correctly")


    def test_load_invalid_data(self):
        """
        Test that invalid data raises a ValueError.
        """
        file_content = ["a a a", "2 4", "3 4 5 6", "# 3 4 6"]
        file_data = "\n".join(file_content)
        mocked_open = my_mock_open(read_data=file_data)

        with patch('builtins.open', mocked_open):
            with patch('numpy.DataSource.open', mocked_open):
                with self.assertRaises(ValueError,
                                       msg="File with no float float float "
                                           "data cannot be loaded"):
                    load_scattering_data("test")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestSasIO("test_parse_3_ok"))
    test_suite.addTest(TestSasIO("test_parse_no_data"))
    test_suite.addTest(TestSasIO("test_parse_no_valid_data"))
    test_suite.addTest(TestSasIO("test_load_clean_data"))
    test_suite.addTest(TestSasIO("test_load_data_with_unescaped_header"))
    test_suite.addTest(TestSasIO("test_load_data_with_unescaped_footer"))
    test_suite.addTest(TestSasIO("test_load_invalid_data"))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
