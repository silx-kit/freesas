#!/usr/bin/python3
# coding: utf-8

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2015, ESRF"
__date__ = "06/02/2026"

import os
import logging
import glob
import platform
from itertools import combinations
from collections import namedtuple
from freesas.cormap import gof
from freesas.sasio import load_scattering_data
from freesas.sas_argparser import SASParser
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cormap")

datum = namedtuple("datum", ["index", "filename", "data"])


operatingSystem = platform.system()



def parse():
    """Parse input and return list of files.
    :return: list of input files
    """
    description = "Measure pair-wise similarity of spectra "
    epilog = """cormapy is an open-source implementation of
    the cormap algorithm in datcmp (from ATSAS).
    It does not scale the data and assume they are already scaled
    """
    parser = SASParser(prog="cormapy", description=description, epilog=epilog)
    parser.add_file_argument(help_text="dat files to compare")

    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    files = [i for i in args.file if os.path.exists(i)]
    if operatingSystem == "Windows" and files == []:
        files = glob.glob(args.file[0])
        files.sort()
    input_len = len(files)
    logger.debug("%s input files" % input_len)
    return files


def compare(lstfiles):
    res = [
        "Pair-wise Correlation Map",
        "" "                                C       Pr(>C)",
    ]
    data = []
    for i, f in enumerate(lstfiles):
        try:
            ary = load_scattering_data(f)
        except ValueError as e:
            print(e)
        if ary.ndim > 1 and ary.shape[1] > 1:
            ary = ary[:, 1]
        d = datum(i + 1, f, ary)
        data.append(d)
    for a, b in combinations(data, 2):
        r = gof(a.data, b.data)
        res.append(
            "%6i vs. %6i          %6i     %8.6f" % (a.index, b.index, r.c, r.P)
        )
    res.append("")
    for a in data:
        res.append(
            "%6i         %8f + %8f * %s" % (a.index, 0.0, 1.0, a.filename)
        )
    res.append("")
    print(os.linesep.join(res))
    return res


def main():
    """main entry point"""
    f = parse()
    if f:
        compare(f)


if __name__ == "__main__":
    main()
