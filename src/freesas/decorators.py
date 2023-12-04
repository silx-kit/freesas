# coding: utf-8
#
#    Project: Free SAS tools
#             https://github.com/kif/freesas
#
#    Copyright (C) 2015-2020 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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
"""Bunch of useful decorators"""

__authors__ = ["Jerome Kieffer", "H. Payno", "P. Knobel", "V. Valls"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/04/2020"
__status__ = "development"
__docformat__ = 'restructuredtext'

import sys
import time
import logging

timelog = logging.getLogger("freesas.timeit")


def timeit(func):

    def wrapper(*arg, **kw):
        '''This is the docstring of timeit:
        a decorator that logs the execution time'''
        t1 = time.perf_counter()
        res = func(*arg, **kw)
        t2 = time.perf_counter()
        name = func.func_name if sys.version_info[0] < 3 else func.__name__
        timelog.warning("%s took %.3fs", name, t2 - t1)
        return res

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
