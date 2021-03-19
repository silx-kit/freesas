#!/usr/bin/python3
# coding: utf-8
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2017-2020  European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2017-2020, ESRF"
__date__ = "05/06/2020"


__author__ = "Jérôme Kieffer, Martha Brennich"
__license__ = "MIT"
__copyright__ = "2021, ESRF"
__date__ = "19/03/2021"

import sys
import logging
from argparse import Namespace
from freesas.autorg import autoRg
from .sas_argparser import GuinierParser
from .guinier_fitting import run_guinier_fit

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("auto_gpa")

if sys.version_info < (3, 6):
    logger.error("This code uses F-strings and requires Python 3.6+")


def parse() -> Namespace:
    """Parse input and return list of files.
    :return: list of input files
    """
    description = (
        "Calculate the radius of gyration using Guinier law"
        " for a set of scattering curves"
    )
    epilog = """free_rg is an open-source implementation of
    the autorg algorithm originately part of the ATSAS suite.
    As this is reverse engineered, some constants and results may differ
    """
    parser = GuinierParser(
        prog="free_rg", description=description, epilog=epilog
    )
    return parser.parse_args()


def main() -> None:
    run_guinier_fit(fit_function=autoRg, parser=parse, logger=logger)


if __name__ == "__main__":
    main()
