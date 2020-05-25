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

"Tool to perform a simple plotting of a set of SAS curve"

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "14/05/2020"

import os
import argparse
import logging
import glob
import platform
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plot_sas")

import numpy
import freesas
from freesas import plot


def parse():
    """ Parse input and return list of files.
    :return: list of input files
    """
    usage = "freesas.py [OPTIONS] FILES "
    description = "Generate typical sas plots with matplotlib"
    epilog = """freesas is an open-source implementation of a bunch of
    small angle scattering algorithms. """
    version = "freesas.py version %s from %s" % (freesas.version, freesas.date)
    parser = argparse.ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("file", metavar="FILE", nargs='+', help="dat files to plot")
    parser.add_argument("-o", "--output", action='store', help="Output filename", default=None, type=str)
    parser.add_argument("-f", "--format", action='store', help="Output format: jpeg, svg, png, pdf", default=None, type=str)
    parser.add_argument("-v", "--verbose", default=False, help="switch to verbose mode", action='store_true')
    parser.add_argument("-V", "--version", action='version', version=version)
    return parser.parse_args()


def main():
    args = parse()
    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    files = [i for i in args.file if os.path.exists(i)]
    if platform.system() == "Windows" and files == []:
        files = glob.glob(args.file[0])
        files.sort()
    input_len = len(files)
    logger.debug("%s input files" % input_len)
    figures = []
    if len(files) > 1 and args.output:
        logger.warning("Only PDF export is possible in multi-frame mode")
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        raise NotImplementedError("TODO")

    for afile in files:
        try:
            data = numpy.loadtxt(afile)
        except:
            logger.error("Unable to parse file %s", afile)
        else:
            fig = plot.plot_all(data, filename=args.output, format=args.format)
            figures.append(fig)
        if args.output is None:
            fig.show()
    if not args.output:
        input("Press enter to quit")


if __name__ == "__main__":
    main()
