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
__date__ = "20/04/2020"

import os
import argparse
import logging
import glob
import platform
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bift")

import numpy
import freesas
from freesas import bift


def parse():
    """ Parse input and return list of files.
    :return: list of input files
    """
    usage = "bift.py [OPTIONS] FILES "
    description = "Calculate the density as function of distance p(r) curve from an I(q) scattering curve"
    epilog = """bift.py is a Python implementation of the Bayesian Inverse Fourier Transform 
    
    This code is the implementation of 
    Steen Hansen J. Appl. Cryst. (2000). 33, 1415-1421

    Based on the BIFT from Jesse Hopkins, available at:
    https://sourceforge.net/p/bioxtasraw/git/ci/master/tree/bioxtasraw/BIFT.py

    It aims at being a drop in replacement for datgnom of the ATSAS suite.

    """
    version = "bift.py version %s from %s" % (freesas.version, freesas.date)
    parser = argparse.ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("file", metavar="FILE", nargs='+', help="dat files to compare")
    parser.add_argument("-v", "--verbose", default=False, help="switch to verbose mode", action='store_true')
    parser.add_argument("-V", "--version", action='version', version=version)
    args = parser.parse_args()
    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    files = [i for i in args.file if os.path.exists(i)]
    if platform.system() == "Windows" and files == []:
        files = glob.glob(args.file[0])
        files.sort()
    input_len = len(files)
    logger.debug("%s input files" % input_len)
    return files


def main():
    list_files = parse()
    for afile in list_files:
        try:
            data = numpy.loadtxt(afile)
        except:
            logger.error("Unable to parse file %s", afile)
        else:
            try:
                bo = bift.auto_bift(data)
            except Exception as err:
                print("%s %s" % (afile, err))
            else:
                stats = bo.calc_stats()
                "radius density_avg density_std evidence_avg evidence_std Dmax_avg Dmax_std alpha_avg, alpha_std chi2_avg chi2_std Rg_avg Rg_std I0_avg I0_std"
                res = ["Dmax= %.2f ±%.2f" % (stats.Dmax_avg, stats.Dmax_std),
                       "𝛂= %.1f ±%.2f" % (stats.alpha_avg, stats.alpha_std),
                       "χ²= %.2f ±%.2f" % (stats.chi2_avg, stats.chi2_std),
                       "S₀= %.4f ±%.4f" % (stats.regularization_avg, stats.regularization_std),
                       "logP= %.2f ±%.4f" % (stats.evidence_avg, stats.evidence_std),
                       "Rg= %.2f ±%.2f" % (stats.Rg_avg, stats.Rg_std),
                       "I₀= %.2f ±%.2f" % (stats.I0_avg, stats.I0_std),
                       ]

                print(afile + ": " + "; ".join(res))
                dest = os.path.splitext(afile)[0] + ".out"
                with open(dest, "wt") as out:
                    out.write("# %s %s" % (afile, os.linesep))
                    for txt in res:
                        out.write("# %s %s" % (txt, os.linesep))
                    out.write("%s# r p(r) sigma_p(r)%s" % (os.linesep, os.linesep))
                    for r, p, s in zip(stats.radius, stats.density_avg, stats.density_std):
                        out.write("%s\t%s\t%s%s" % (r, p, s, os.linesep))


if __name__ == "__main__":
    main()
