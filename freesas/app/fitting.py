"""This module provides a function which reads in the data,
performs the guinier fit with a given algotithm and reates the input."""

import sys
import logging
import platform
from os import linesep as os_linesep
from pathlib import Path
from typing import Callable, List, Optional, IO
from numpy import ndarray
from freesas.autorg import (
    RG_RESULT,
    InsufficientDataError,
    NoGuinierRegionError,
)
from freesas.sasio import (
    load_scattering_data,
    convert_inverse_angstrom_to_nanometer,
)
from .sas_argparser import GuinierParser


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


def get_output_destination(output_path: Optional[Path]) -> IO[str]:
    """
    Return file or stdout object to write output to
    :param output_path: None if output to stdout, else Path to outputfile
    """
    # pylint: disable=R1705
    if output_path is not None:
        return open(output_path, "w")
    else:
        return sys.stdout


def get_linesep(output_destination: IO[str]) -> str:
    """
    Get the appropriate linesep depending on the output destination.
    :param output_destination: an IO object, e.g. an open file or stdout
    """
    # pylint: disable=R1705
    if output_destination == sys.stdout:
        return os_linesep
    else:
        return "\n"


def get_header(output_format: str, linesep: str) -> str:
    """Return appropriate header line for selected output format"""
    # pylint: disable=R1705
    if output_format == "csv":
        return (
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
        return ""


def run_guinier_fit(
    fit_function: Callable[[ndarray], RG_RESULT],
    parser: GuinierParser,
    logger: logging.Logger,
) -> None:
    """
    reads in the data, performs the guinier fit with a given algotithm and
    creates the
    :param fit_function : A Guinier fit function data -> RG_RESULT
    :param parser: a function that returns the output of argparse.parse()
    :param logger: a Logger
    """
    args = parser.parse_args()
    set_logging_level(args.verbose)
    files = collect_files(args.file)
    logger.debug("%s input files", len(files))

    output_destination = get_output_destination(args.output)
    linesep = get_linesep(output_destination)

    output_destination.write(get_header(args.format, linesep))

    for afile in files:
        logger.info("Processing %s", afile)
        try:
            data = load_scattering_data(afile)
        except OSError:
            logger.error("Unable to read file %s", afile)
        except ValueError:
            logger.error("Unable to parse file %s", afile)
        else:
            if args.unit == "Å":
                data = convert_inverse_angstrom_to_nanometer(data)
            try:
                rg = fit_function(data)
            except (
                InsufficientDataError,
                NoGuinierRegionError,
                ValueError,
                IndexError,
            ) as err:
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
                output_destination.write(res)
                output_destination.write(linesep)
                output_destination.flush()
    if output_destination is not sys.stdout:
        output_destination.close()
