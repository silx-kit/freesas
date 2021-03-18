"""This module provides a function which reads in the data,
performs the guinier fit with a given algotithm and reates the input."""

import sys
import logging
import platform
from os import linesep as os_linesep
from pathlib import Path
from typing import Callable
from argparse import Namespace
from numpy import ndarray
from freesas.autorg import RG_RESULT
from freesas.sasio import (
    load_scattering_data,
    convert_inverse_angstrom_to_nanometer,
)


def run_guinier_fit(
    fit_function: Callable[[ndarray], RG_RESULT],
    parser: Callable[[], Namespace],
    logger: logging.Logger,
) -> None:
    """
    reads in the data, performs the guinier fit with a given algotithm and
    creates the
    :param fit_function : A Guinier fit function data -> RG_RESULT
    :param parser: a function that returns the output of argparse.parse()
    :param logger: a Logger
    """
    args = parser()
    if args.verbose == 1:
        logging.root.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logging.root.setLevel(logging.DEBUG)
    files = [Path(i) for i in args.file if Path(i).exists()]
    if platform.system() == "Windows" and files == []:
        files = list(Path.cwd().glob(args.file[0]))
        files.sort()
    input_len = len(files)
    logger.debug("%s input files", input_len)

    if args.output:
        dst = open(args.output, "w")
        linesep = "\n"
    else:
        dst = sys.stdout
        linesep = os_linesep

    if args.format == "csv":
        dst.write(
            "File,Rg,Rg StDev,I(0),I(0) StDev,First point,"
            "Last point,Quality,Aggregated" + linesep
        )

    for afile in files:
        logger.info("Processing %s", afile)
        try:
            data = load_scattering_data(afile)
        except:
            logger.error("Unable to parse file %s", afile)
        else:
            if args.unit == "Å":
                data = convert_inverse_angstrom_to_nanometer(data)
            try:
                rg = fit_function(data)
            except Exception as err:
                sys.stdout.write(
                    "%s, %s: %s\n" % (afile, err.__class__.__name__, err)
                )
            else:
                if args.format == "csv":
                    res = f"{afile},{rg.Rg:6.4f},{rg.sigma_Rg:6.4f},{rg.I0:6.4f},{rg.sigma_I0:6.4f},{rg.start_point:3},{rg.end_point:3},{rg.quality:6.4f},{rg.aggregated:6.4f}"
                elif args.format == "ssv":
                    res = f"{rg.Rg:6.4f} {rg.sigma_Rg:6.4f} {rg.I0:6.4f} {rg.sigma_I0:6.4f} {rg.start_point:3} {rg.end_point:3} {rg.quality:6.4f} {rg.aggregated:6.4f} {afile}"
                else:
                    res = "%s %s" % (afile, rg)
                dst.write(res)
                dst.write(linesep)
                dst.flush()
    if args.output:
        dst.close()
