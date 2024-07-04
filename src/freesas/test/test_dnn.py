# -*- coding: utf-8 -*-
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2024-2024  European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "03/07/2024"

import unittest
import logging
import os
from .utilstest import get_datafile
from ..resources import resource_filename
from ..sasio import load_scattering_data
from ..dnn import preprocess
logger = logging.getLogger(__name__)

class TestDNN(unittest.TestCase):

    def test_preprocess(self):
        """
        Test for the preprocessing function 
        """
        datfile = get_datafile("bsa_005_sub.dat")
        data = load_scattering_data(datfile)
        q, I, sigma = data.T
        Iprep = preprocess(q, I)
        self.assertEqual(Iprep.max(), 1, msg="range 0-1")
        self.assertEqual(Iprep.shape, (1024,), msg="size 1024")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestDNN("test_preprocess"))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
