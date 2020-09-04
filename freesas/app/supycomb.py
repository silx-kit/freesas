#!/usr/bin/python3
__author__ = "Guillaume Bonamis"
__license__ = "MIT"
__copyright__ = "2015, ESRF"
__date__ = "09/07/2020"

import logging
from os.path import dirname, abspath
from freesas.align import InputModels, AlignModels
from .sas_argparser import SASParser

base = dirname(dirname(abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supycomb")

def parse():

    """ Parse input and return list of files.
    :return: list of args
    """
    description = "Align several models and calculate NSD"
    epilog = """supycomb is an open-source implementation of
    [J. Appl. Cryst. (2001). 34, 33-41](doi:10.1107/S0021889800014126).

    The main difference with supcomb: the fast mode does not re-bin beads. It only refines the best matching orientation which provides a speed-up of a factor 8.

    """
    parser = SASParser(prog="supycomp.py", description=description, epilog=epilog)
    parser.add_file_argument(help_text="pdb files to align")
    parser.add_argument("-m", "--mode", dest="mode", type=str, choices=["SLOW", "FAST"], default="SLOW", help="Either SLOW or FAST, default: %(default)s)")
    parser.add_argument("-e", "--enantiomorphs", type=str, choices=["YES", "NO"], default="YES", help="Search enantiomorphs, YES or NO, default: %(default)s)")
    parser.add_argument("-q", "--quiet", type=str, choices=["ON", "OFF"], default="ON", help="Hide log or not, default: %(default)s")
    parser.add_argument("-g", "--gui", type=str, choices=["YES", "NO"], default="YES", help="Use GUI for figures or not, default: %(default)s")
    parser.add_argument("-o", "--output", type=str, default="aligned.pdb", help="output filename, default: %(default)s")
    return parser.parse_args()


def main():
    """main application"""

    args = parse()
    input_len = len(args.file)
    logger.info("%s input files" % input_len)
    selection = InputModels()

    if args.mode == "SLOW":
        slow = True
        logger.info("SLOW mode")
    else:
        slow = False
        logger.info("FAST mode")

    if args.enantiomorphs == "YES":
        enantiomorphs = True
    else:
        enantiomorphs = False
        logger.info("NO enantiomorphs")

    if args.quiet == "OFF":
        logger.setLevel(logging.DEBUG)
        logger.info("setLevel: Debug")

    if args.gui == "NO":
        save = True
        logger.info("Figures saved automatically : \n  R factor values and selection =>  Rfactor.png \n  NSD table and selection =>  nsd.png")
    else:
        save = False

    align = AlignModels(args.file, slow=slow, enantiomorphs=enantiomorphs)
    if input_len == 2:
        align.outputfiles = args.output
        align.assign_models()
        dist = align.alignment_2models()
        logger.info("%s and %s aligned" % (args.file[0], args.file[1]))
        logger.info("NSD after optimized alignment = %.2f" % dist)
    else:
        align.outputfiles = ["model-%02i.pdb" % (i + 1) for i in range(input_len)]
        selection.inputfiles = args.file
        selection.models_selection()
        selection.rfactorplot(save=save)
        align.models = selection.sasmodels
        align.validmodels = selection.validmodels

        align.makeNSDarray()
        align.alignment_reference()
        logger.info("valid models aligned on the model %s" % (align.reference + 1))
        align.plotNSDarray(rmax=round(selection.rmax, 4), save=save)

    if not save and input_len > 2:
        input("Press any key to exit")


if __name__ == "__main__":
    main()
