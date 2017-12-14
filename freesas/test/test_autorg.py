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
__date__ = "07/09/2017"

import numpy
import unittest
from .utilstests import get_datafile
from ..autorg import autoRg, RG_RESULT, linFit
import logging
logger = logging.getLogger(__name__)


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
        logger.debug("Reference version: %s" % atsas_result.pop("Version"))
        atsas_result = RG_RESULT(**atsas_result)
        free_result = autoRg(data)
        logger.debug("Ref: %s" % (atsas_result,))
        logger.debug("Obt: %s" % (free_result,))
        self.assertAlmostEqual(atsas_result.Rg, free_result.Rg, 1, "RG fits within 2 digits")
        self.assertAlmostEqual(atsas_result.I0, free_result.I0, msg="I0 fits within +/- 1 ", delta=1)
        self.assertAlmostEqual(atsas_result.quality, free_result.quality, 1, msg="quality fits within 1 digits")

class TestFit(unittest.TestCase): 
    #Testcase originally comes from wikipedia article on linear regression, expected results from scipy.stats.linregress
    testx = [1.47,1.5,1.52,1.55, 1.57,1.6,1.63,1.65,1.68,1.7,1.73,1.75,1.78,1.80,1.83]
    testy = [52.21,53.12,54.48,55.84,57.20,58.57,59.93,61.29,63.11,64.47,66.28,68.1,69.92,72.19,74.46]
    testw = [1.0] *15
    testintercept = -39.061956
    testslope = -61.2721865

    def test_linFit(self):
        print("Testing Linear Fitting")
        #fit_result = self.atsas_autorg.copy()
        #logger.debug("Reference version: %s" % atsas_result.pop("Version"))
        #atsas_result = RG_RESULT(**atsas_result)
        #free_result = autoRg(data)
        fit_result = linFit(self.testx,self.testy,self.testw)
        #print(fit_result)
        self.assertAlmostEqual(fit_result[0], self.testintercept, 5,"Intercept fits wihtin 4(?) digits")    
        self.assertAlmostEqual(fit_result[1], self.testslope, 5,"Intercept fits wihtin 4(?) digits") 

def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAutoRg("test_autorg"))
    testSuite.addTest(TestFit("test_linFit"))
    return testSuite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
