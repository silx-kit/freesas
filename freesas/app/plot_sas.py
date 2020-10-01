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

import platform
import logging
from pathlib import Path
from matplotlib.pyplot import switch_backend
from matplotlib.backends.backend_pdf import PdfPages
from freesas import plot
from freesas.sasio import (
    load_scattering_data,
    convert_inverse_angstrom_to_nanometer,
)
from freesas.autorg import InsufficientDataError, NoGuinierRegionError
from .sas_argparser import SASParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plot_sas")


def set_backend(output: Path = None, outputformat: str = None):
    """Explicitely set silent backend based on format or filename
    Needed on MacOS
    @param output: Name of the specified output file
    @param format: User specified format
    """
    if outputformat:
        outputformat = outputformat.lower()
    elif output and len(output.suffix) > 0:
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
    """Parse input and return list of files.
    :return: list of input files
    """
    description = "Generate typical sas plots with matplotlib"
    epilog = """freesas is an open-source implementation of a bunch of
    small angle scattering algorithms. """
    parser = SASParser(
        prog="freesas.py", description=description, epilog=epilog
    )
    parser.add_file_argument(help_text="dat files to plot")
    parser.add_output_filename_argument()
    parser.add_output_data_format("jpeg", "svg", "png", "pdf")
    parser.add_q_unit_argument()
    return parser.parse_args()


def create_figure(file: Path, unit: str = "nm"):
    """Create multi-plot SAS figure for data from a file
    @param file: filename of SAS file in q I Ierr format
    @param unit: length unit of input data, supported options are Å and nm.
    :return: figure with SAS plots for this file
    """
    data = load_scattering_data(file)
    if unit == "Å":
        data = convert_inverse_angstrom_to_nanometer(data)
    fig = plot.plot_all(data)
    fig.suptitle(file)
    return fig


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

    if args.output and len(files) > 1:
        logger.warning("Only PDF export is possible in multi-frame mode")
    if args.output and platform.system() == "Darwin":
        if len(files) == 1:
            set_backend(args.output, args.format)
        elif len(files) > 1:
            set_backend(outputformat="pdf")
    for afile in files:
        try:
            fig = create_figure(afile, args.unit)
        except OSError:
            logger.error("Unable to load file %s", afile)
        except (InsufficientDataError, NoGuinierRegionError, ValueError):
            logger.error("Unable to process file %s", afile)
        else:
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
