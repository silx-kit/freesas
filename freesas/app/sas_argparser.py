# -*- coding: utf-8 -*-
"""
Generalized arg parser for freeSAS apps to ensure unified command line API.
"""

__author__ = "Martha Brennich"
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "09/08/2020"

import argparse
from pathlib import Path
from freesas import dated_version as freesas_version

def parse_unit(unit_input: str) -> str:
    """
    Parser for sloppy acceptance of unit flags.
    Current rules:
    "A" ➔ "Å"
    :param unit_input: unit flag as provided by the user
    :return: cast of user input to known flag if sloppy rule defined,
             else user input.
    """
    if unit_input == "A": # pylint: disable=R1705
        return "Å"
    else:
        return unit_input

class SASParser:

    """
    Wrapper class for argparse ArgumentParser that provides predefined argument.
    """

    def __init__(self, prog: str, description: str, epilog: str, **kwargs):
        """
        Create parser argparse ArgumentParser
        - standardized usage text
        - standardized verion text
        - verbose and version args added by default

        :param prog:         name of the executable
        :param description:  description param of argparse ArgumentParser
        :param epilog:       epilog param of argparse ArgumentParser
        :param kwargs:       additional kwargs for argparse ArgumentParser
        """

        usage = "%s [OPTIONS] FILES " %(prog)
        version = "%s version %s from %s" %(prog, freesas_version.version,
                                            freesas_version.date)

        self.parser = argparse.ArgumentParser(usage=usage,
                                              description=description,
                                              epilog=epilog, **kwargs)
        self.add_argument("-v", "--verbose", default=0,
                          help="switch to verbose mode", action='count')
        self.add_argument("-V", "--version", action='version', version=version)

    def parse_args(self):
        """ Wrapper for argparse parse_args() """
        return self.parser.parse_args()

    def add_argument(self, *args, **kwargs):
        """ Wrapper for argparse add_argument() """
        self.parser.add_argument(*args, **kwargs)

    def add_file_argument(self, help_text: str):
        """
        Add positional file argument.

        :param help_text: specific help text to be displayed
        """
        self.add_argument("file", metavar="FILE", nargs='+',
                          help=help_text)


    def add_q_unit_argument(self):
        """
        Add default argument for selecting length unit of input data
        between Å and nm. nm is default.
        """
        self.add_argument("-u", "--unit", action='store',
                          choices=["nm", "Å", "A"],
                          help="Unit for q: inverse nm or Ångstrom?",
                          default="nm", type=parse_unit)

    def add_output_filename_argument(self):
        """ Add default argument for specifying output format. """
        self.add_argument("-o", "--output", action='store',
                          help="Output filename", default=None, type=Path)

    def add_output_data_format(self, *formats: str, default: str = None):
        """ Add default argument for specifying output format. """
        help_string = "Output format: " +  ", ".join(formats)
        self.add_argument("-f", "--format", action='store',
                          help=help_string, default=default, type=str)

class GuinierParser:
    """
    Wrapper class for argparse ArgumentParser that provides predefined
    arguments for auto_rg like programs.
    """

    def __init__(self, prog: str, description: str, epilog: str, **kwargs):
        """
        Create parser argparse ArgumentParser with argument
        - standardized usage text
        - standardized version text
        - verbose and version args added by default

        :param prog:         name of the executable
        :param description:  description param of argparse ArgumentParser
        :param epilog:       epilog param of argparse ArgumentParser
        :param kwargs:       additional kwargs for argparse ArgumentParser
        """

        file_help_text = "dat files of the scattering curves"
        self.parser = SASParser(prog=prog, description=description,
                                epilog=epilog, **kwargs)
        self.parser.add_file_argument(help_text=file_help_text)
        self.parser.add_output_filename_argument()
        self.parser.add_output_data_format("native", "csv", "ssf",
                                           default="native")
        self.parser.add_q_unit_argument()

    def parse_args(self):
        """ Wrapper for SASParser parse_args() """
        return self.parser.parse_args()

    def add_argument(self, *args, **kwargs):
        """ Wrapper for SASParser add_argument() """
        self.parser.add_argument(*args, **kwargs)
