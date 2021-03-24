#!/usr/bin/python3
# coding: utf-8
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2020  European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = ["Jérôme Kieffer", "Martha Brennich"]
__license__ = "MIT"
__copyright__ = "2021, ESRF"
__date__ = "19/03/2021"

import sys
import logging
from freesas.autorg import auto_guinier
from freesas.sas_argparser import GuinierParser
from freesas.fitting import run_guinier_fit

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("auto_guinier")

if sys.version_info < (3, 6):
    logger.error("This code uses F-strings and requires Python 3.6+")


def build_parser() -> GuinierParser:
    """Build parser for input and return list of files.
    :return: parser
    """
    description = (
        "Calculate the radius of gyration using linear fitting of"
        "logarithmic intensities for a set of scattering curves"
    )
    epilog = """free_guinier is an open-source implementation of
    the autorg algorithm originately part of the ATSAS suite.
    As this tool used a different theory, some results may differ
    """
    return GuinierParser(
        prog="free_guinier", description=description, epilog=epilog
    )


def main() -> None:
    """Entry point for free_guinier app"""
    parser = build_parser()
    run_guinier_fit(fit_function=auto_guinier, parser=parser, logger=logger)


if __name__ == "__main__":
    main()
