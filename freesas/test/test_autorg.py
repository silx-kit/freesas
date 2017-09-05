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

__authors__ = ["J. Kieffer"]
__license__ = "MIT"
__date__ = "05/09/2017"

import numpy
import unittest
from .utilstests import get_datafile
from ..autorg import autoRg, RG_RESULT


class TestAutoRg(unittest.TestCase):
    testfile = get_datafile("bsa_005_sub.dat")
    # Reference implementation
    atsas_autorg = {"Version": "Atsas 2.6.1",
                    "Rg": 2.98016,
                    "sigma_Rg": 0.156859,
                    "I0": 61.3093,
                    "sigma_I0": 0.0606315,
                    "start_point": 46,
                    "end_point": 95,
                    "quality": 0.752564,
                    "aggregated": 0}

    def test_autorg(self):
        print(self.testfile)
        data = numpy.loadtxt(self.testfile)
        atsas_result = self.atsas_autorg.copy()
        print("Reference version: %s" % atsas_result.pop("Version"))
        print("Ref: %s" % (RG_RESULT(**atsas_result),))
        print("Obt: %s" % (autoRg(data),))
        print()


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAutoRg("test_autorg"))
    return testSuite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
