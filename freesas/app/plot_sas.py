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

import argparse
import platform
import logging
from pathlib import Path
from matplotlib.pyplot import switch_backend
from matplotlib.backends.backend_pdf import PdfPages
from freesas import dated_version as freesas_version
from freesas import plot
from freesas.sasio import load_scattering_data
from freesas.autorg import InsufficientDataError, NoGuinierRegionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plot_sas")

def set_backend(output: Path = None, outputformat: str = None):
    """ Explicitely set silent backend based on format or filename
        Needed on MacOS
        @param output: Name of the specified output file
        @param format: User specified format
    """
    if outputformat:
        outputformat = outputformat.lower()
    elif len(output.suffix) > 0:
        outputformat = output.suffix.lower()[1:]
    if outputformat:
        if outputformat == "svg":
            switch_backend("svg")
        elif outputformat in ["ps", "eps"]:
            switch_backend("ps")
        elif outputformat == "pdf":
            switch_backend("pdf")
        elif outputformat == "png":
            switch_backend("agg")

def parse():
    """ Parse input and return list of files.
    :return: list of input files
    """
    usage = "freesas.py [OPTIONS] FILES "
    description = "Generate typical sas plots with matplotlib"
    epilog = """freesas is an open-source implementation of a bunch of
    small angle scattering algorithms. """
    version = "freesas.py version %s from %s" % (freesas_version.version,
                                                 freesas_version.date)
    parser = argparse.ArgumentParser(usage=usage,
                                     description=description,
                                     epilog=epilog)
    parser.add_argument("file", metavar="FILE", nargs='+',
                        help="dat files to plot")
    parser.add_argument("-o", "--output", action='store',
                        help="Output filename", default=None, type=Path)
    parser.add_argument("-f", "--format", action='store',
                        help="Output format: jpeg, svg, png, pdf",
                        default=None, type=str)
    parser.add_argument("-v", "--verbose", default=False,
                        help="switch to verbose mode", action='store_true')
    parser.add_argument("-V", "--version", action='version', version=version)
    return parser.parse_args()


def main():
    args = parse()
    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    files = [Path(i) for i in args.file if Path(i).exists()]
    if platform.system() == "Windows" and files == []:
        files = list(Path.cwd().glob(args.file[0]))
        files.sort()
    input_len = len(files)
    logger.debug("%s input files", input_len)
    figures = []

    if args.output and platform.system() == "Darwin":
        if len(files) == 1:
            set_backend(args.output, args.format)
        elif len(files) > 1:
            logger.warning("Only PDF export is possible in multi-frame mode")
            set_backend(outputformat="pdf")
    for afile in files:
        try:
            data = load_scattering_data(afile)
        except OSError:
            logger.error("Unable to load file %s", afile)
        except ValueError:
            logger.error("Unable to parse file %s", afile)
        else:
            try:
                fig = plot.plot_all(data)
            except (InsufficientDataError, NoGuinierRegionError, ValueError):
                logger.error("Unable to process file %s", afile)
            else:
                fig.suptitle(afile)
                figures.append(fig)
                if args.output is None:
                    fig.show()
                elif len(files) == 1:
                    fig.savefig(args.output, format=args.format)
    if len(figures) > 1 and args.output:
        with PdfPages(args.output) as pdf_output_file:
            for fig in figures:
                pdf_output_file.savefig(fig)
    if not args.output:
        input("Press enter to quit")



if __name__ == "__main__":
    main()
