__author__ = "Guillaume Bonamis"
__license__ = "MIT"
__copyright__ = "2015, ESRF"
__date__ = "09/07/2020"

import logging
from os.path import abspath, dirname

from freesas.align import AlignModels, InputModels
from freesas.sas_argparser import (
    SASParser,
    check_output_template,
    format_output_filename,
)

base = dirname(dirname(abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supycomb")

#: Default output templates, for two models and for more
DEFAULT_OUTPUT_TEMPLATE = "aligned.pdb"
DEFAULT_MULTI_OUTPUT_TEMPLATE = "model-{index:02d}.pdb"


def parse():
    """Parse input and return list of files.
    :return: list of args
    """
    description = "Align several models and calculate NSD"
    epilog = """supycomb is an open-source implementation of
    [J. Appl. Cryst. (2001). 34, 33-41](doi:10.1107/S0021889800014126).

    The main difference with supcomb: the fast mode does not re-bin beads. It only refines the best matching orientation which provides a speed-up of a factor 8.

    """
    parser = SASParser(prog="supycomp", description=description, epilog=epilog)
    parser.add_file_argument(help_text="pdb files to align")
    parser.add_argument(
        "-m",
        "--mode",
        dest="mode",
        type=str,
        choices=["SLOW", "FAST"],
        default="SLOW",
        help="Either SLOW or FAST, default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--enantiomorphs",
        type=str,
        choices=["YES", "NO"],
        default="YES",
        help="Search enantiomorphs, YES or NO, default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        type=str,
        choices=["ON", "OFF"],
        default="ON",
        help="Hide log or not, default: %(default)s",
    )
    parser.add_argument(
        "-g",
        "--gui",
        type=str,
        choices=["YES", "NO"],
        default="YES",
        help="Use GUI for figures or not, default: %(default)s",
    )
    # the default depends on the number of input models, main() picks it
    parser.add_output_template_argument(
        None,
        extra_fields={"index": "number of the model, starting at 1"},
        described_default=(
            f"{DEFAULT_OUTPUT_TEMPLATE} for two models, "
            f"{DEFAULT_MULTI_OUTPUT_TEMPLATE} for more"
        ),
    )
    return parser.parse_args()


def main():
    """main application"""

    args = parse()
    input_len = len(args.file)
    logger.info(f"{input_len} input files")
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
        logger.info(
            "Figures saved automatically : \n  R factor values and selection =>  Rfactor.png \n  NSD table and selection =>  nsd.png"
        )
    else:
        save = False

    align = AlignModels(args.file, slow=slow, enantiomorphs=enantiomorphs)
    if input_len == 2:
        template = args.output or DEFAULT_OUTPUT_TEMPLATE
        check_output_template(template, index=0)
        # only the second model gets aligned onto the first one and saved
        align.outputfiles = str(
            format_output_filename(template, args.file[1], mkdir=True, index=2)
        )
        align.assign_models()
        dist = align.alignment_2models()
        logger.info(f"{args.file[0]} and {args.file[1]} aligned")
        logger.info(f"NSD after optimized alignment = {dist:.2f}")
    else:
        template = args.output or DEFAULT_MULTI_OUTPUT_TEMPLATE
        check_output_template(template, index=0)
        if "{index" not in template and "{basename" not in template:
            logger.warning(
                "Output template %s has neither {index} nor {basename} field: "
                "every aligned model will be written to the same file",
                template,
            )
        align.outputfiles = [
            str(format_output_filename(template, afile, mkdir=True, index=idx + 1))
            for idx, afile in enumerate(args.file)
        ]
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
