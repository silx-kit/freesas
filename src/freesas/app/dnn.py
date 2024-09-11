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

__author__ = ["Jérôme Kieffer", "Mayank Yadav"]
__license__ = "MIT"
__copyright__ = "2024, ESRF"
__date__ = "11/09/2024"

import sys
import logging
from freesas.sas_argparser import SASParser
from freesas.fitting import run_dnn

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("free_dnn")

if sys.version_info < (3, 6):
    logger.error("This code uses F-strings and requires Python 3.6+")


def build_parser() -> SASParser:
    """Build parser for input and return list of files.
    :return: parser
    """
    description = (
        "Assess the radius of gyration (Rg) and the diameter of the particle (Dmax) using a Dense Neural-Network"
        " for a set of scattering curves"
    )
    epilog = """free_dnn is an alternative implementation of
    `gnnom` (https://doi.org/10.1016/j.str.2022.03.011).
    As this tool used a different training set, some results are likely to differ.
    """
    parser = SASParser(prog="free_gpa", description=description, epilog=epilog)
    file_help_text = "dat files of the scattering curves"
    parser.add_file_argument(help_text=file_help_text)
    parser.add_output_filename_argument()
    parser.add_output_data_format("native", "csv", "ssf", default="native")
    parser.add_q_unit_argument()

    return parser



def main() -> None:
    """Entry point for free_gpa app"""
    parser = build_parser()
    run_dnn(parser=parser, logger=logger)


if __name__ == "__main__":
    main()
