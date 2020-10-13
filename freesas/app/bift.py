#!/usr/bin/python3
# coding: utf-8
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

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2017, ESRF"
__date__ = "13/10/2020"

import sys
import logging
import platform
import traceback
from os import linesep
from pathlib import Path
from numpy import float32
from freesas import bift
from freesas.sasio import load_scattering_data, \
                          convert_inverse_angstrom_to_nanometer
from .sas_argparser import SASParser

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("bift")

def parse():
    """ Parse input and return list of files.
    :return: list of input files
    """
    description = "Calculate the density as function of distance p(r)"\
                  " curve from an I(q) scattering curve"
    epilog = """bift.py is a Python implementation of the Bayesian Inverse Fourier Transform

    This code is the implementation of
    Steen Hansen J. Appl. Cryst. (2000). 33, 1415-1421

    Based on the BIFT from Jesse Hopkins, available at:
    https://sourceforge.net/p/bioxtasraw/git/ci/master/tree/bioxtasraw/BIFT.py

    It aims at being a drop in replacement for datgnom of the ATSAS suite.

    """
    parser = SASParser(prog="bift.py", description=description, epilog=epilog)
    parser.add_file_argument(help_text="I(q) files to convert into p(r)")
    parser.add_output_filename_argument()
    parser.add_q_unit_argument()
    parser.add_argument("-n", "--npt", default=100, type=int,
                        help="number of points in p(r) curve")
    parser.add_argument("-s", "--scan", default=27, type=int,
                        help="Initial alpha-scan size to guess the start parameter")
    parser.add_argument("-m", "--mc", default=100, type=int,
                        help="Number of Monte-Carlo samples in post-refinement")
    parser.add_argument("-t", "--threshold", default=2.0, type=float,
                        help="Sample at average ± threshold*sigma in MC")

    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    files = [Path(i) for i in args.file if Path(i).exists()]
    if platform.system() == "Windows" and files == []:
        files = list(Path.cwd().glob(args.file[0]))
        files.sort()
    input_len = len(files)
    logger.debug("%s input files" % input_len)
    return files, args


def main():
    list_files, args = parse()
    for afile in list_files:
        try:
            data = load_scattering_data(afile)
        except:
            logger.error("Unable to parse file %s", afile)
        else:
            if args.unit == "Å":
                data = convert_inverse_angstrom_to_nanometer(data)
            try:
                bo = bift.auto_bift(data, npt=args.npt, scan_size=args.scan)
            except Exception as err:
                print("%s: %s %s" % (afile, err.__class__.__name__, err))
                if logging.root.level < logging.WARNING:
                    traceback.print_exc(file=sys.stdout)
            else:
                try:
                    stats = bo.monte_carlo_sampling(args.mc, args.threshold,
                                                    npt=args.npt)
                except RuntimeError as err:
                    print("%s: %s %s" % (afile, err.__class__.__name__, err))
                    if logging.root.level < logging.WARNING:
                        traceback.print_exc(file=sys.stdout)
                else:
                    dest = afile.stem + ".out"
                    print(stats.save(dest, source=afile))


if __name__ == "__main__":
    main()
