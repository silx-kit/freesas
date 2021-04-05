"""This module provides a function which reads in the data,
performs the guinier fit with a given algotithm and reates the input."""

__authors__ = ["Martha Brennich"]
__contact__ = "martha.brennich@googlemail.com"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "21/03/2021"
__status__ = "development"
__docformat__ = "restructuredtext"

import sys
import logging
import platform
from os import linesep as os_linesep
from pathlib import Path
from contextlib import contextmanager
from typing import Callable, List, Optional, IO, Generator
from numpy import ndarray
from .autorg import (
    RG_RESULT,
    InsufficientDataError,
    NoGuinierRegionError,
)
from .sasio import (
    load_scattering_data,
    convert_inverse_angstrom_to_nanometer,
)
from .sas_argparser import GuinierParser


def set_logging_level(verbose_flag: int) -> None:
    """
    Set logging level according to verbose flag of argparser
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
    :return: A list of Path objects which includes only existing files
    """
    files = [Path(i) for i in file_list if Path(i).exists()]
    if platform.system() == "Windows" and files == []:
        files = list(Path.cwd().glob(file_list[0]))
        files.sort()
    return files


@contextmanager
def get_output_destination(
    output_path: Optional[Path] = None,
) -> Generator[IO[str], None, None]:
    """
    Return file or stdout object to write output to
    :param output_path: None if output to stdout, else Path to outputfile
    :return: opened file with write access or sys.stdout
    """
    # pylint: disable=R1705
    if output_path is not None:
        with open(output_path, "w") as destination:
            yield destination
    else:
        yield sys.stdout


def get_linesep(output_destination: IO[str]) -> str:
    """
    Get the appropriate linesep depending on the output destination.
    :param output_destination: an IO object, e.g. an open file or stdout
    :return: string with the correct linesep
    """
    # pylint: disable=R1705
    if output_destination == sys.stdout:
        return os_linesep
    else:
        return "\n"


def get_header(linesep: str, output_format: Optional[str] = None) -> str:
    """Return appropriate header line for selected output format
    :param output_format: output format from string parser
    :param linesep: correct linesep for chosen destination
    :return: a one-line string"""
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


def rg_result_to_output_line(
    rg_result: RG_RESULT,
    afile: Path,
    linesep: str,
    output_format: Optional[str] = None,
) -> str:
    """Return result line formatted according to selected output format
    :param rg_result: Result of an rg fit
    :param afile: The name of the file that was processed
    :param output_format: The chosen output format
    :param linesep: correct linesep for chosen destination
    :return: a one-line string including linesep"""
    # pylint: disable=R1705
    if output_format == "csv":
        return (
            ",".join(
                [
                    f"{afile}",
                    f"{rg_result.Rg:6.4f}",
                    f"{rg_result.sigma_Rg:6.4f}",
                    f"{rg_result.I0:6.4f}",
                    f"{rg_result.sigma_I0:6.4f}",
                    f"{rg_result.start_point:3}",
                    f"{rg_result.end_point:3}",
                    f"{rg_result.quality:6.4f}",
                    f"{rg_result.aggregated:6.4f}",
                ]
            )
            + linesep
        )
    elif output_format == "ssv":
        return (
            " ".join(
                [
                    f"{rg_result.Rg:6.4f}",
                    f"{rg_result.sigma_Rg:6.4f}",
                    f"{rg_result.I0:6.4f}",
                    f"{rg_result.sigma_I0:6.4f}",
                    f"{rg_result.start_point:3}",
                    f"{rg_result.end_point:3}",
                    f"{rg_result.quality:6.4f}",
                    f"{rg_result.aggregated:6.4f}",
                    f"{afile}",
                ]
            )
            + linesep
        )
    else:
        return f"{afile} {rg_result}{linesep}"


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

    with get_output_destination(args.output) as output_destination:
        linesep = get_linesep(output_destination)

        output_destination.write(
            get_header(
                linesep,
                args.format,
            )
        )

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
                    rg_result = fit_function(data)
                except (
                    InsufficientDataError,
                    NoGuinierRegionError,
                    ValueError,
                    IndexError,
                ) as err:
                    sys.stderr.write(
                        f"{afile}, {err.__class__.__name__}: {err} {os_linesep}"
                    )
                else:
                    res = rg_result_to_output_line(
                        rg_result,
                        afile,
                        linesep,
                        args.format,
                    )
                    output_destination.write(res)
                    output_destination.flush()
