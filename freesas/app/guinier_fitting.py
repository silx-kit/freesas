"""This module provides a function which reads in the data,
performs the guinier fit with a given algotithm and reates the input."""

import sys
import logging
import platform
from os import linesep as os_linesep
from pathlib import Path
from typing import Callable, List
from argparse import Namespace
from numpy import ndarray
from freesas.autorg import RG_RESULT
from freesas.sasio import (
    load_scattering_data,
    convert_inverse_angstrom_to_nanometer,
)


def set_logging_level(verbose_flag: int) -> None:
    """
    Set logging level according to verbose flaf of argparser
    :param verbose_flag: int flag for logging level
    """
    if verbose_flag == 1:
        logging.root.setLevel(logging.INFO)
    elif verbose_flag >= 2:
        logging.root.setLevel(logging.DEBUG)


def collect_files(file_list: List[str]) -> List[Path]:
    """
    Take file list from argparser and return list of paths
    :param file_list: file list as returned by the argparser
    """
    files = [Path(i) for i in file_list if Path(i).exists()]
    if platform.system() == "Windows" and files == []:
        files = list(Path.cwd().glob(file_list[0]))
        files.sort()
    return files


def get_header(format: str, linesep: str) -> str:
    """Return appropriate header line for selected output format"""
    if format == "csv":
        header = (
            ",".join(
                (
                    "File",
                    "Rg",
                    "Rg StDev",
                    "I(0)",
                    "I(0) StDev",
                    "First point",
                    "Last point",
                    "Quality,Aggregated",
                )
            )
            + linesep
        )
    else:
        header = ""
    return header


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
    set_logging_level(args.verbose)
    files = collect_files(args.file)
    logger.debug("%s input files", len(files))

    if args.output:
        dst = open(args.output, "w")
        linesep = "\n"
    else:
        dst = sys.stdout
        linesep = os_linesep

    dst.write(get_header(args.format, linesep))

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
