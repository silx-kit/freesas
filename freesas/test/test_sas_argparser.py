#!/usr/bin/python
# coding: utf-8

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "25/03/2022"


import unittest
import logging
import freesas
from freesas import dated_version as freesas_version
from freesas.sas_argparser import SASParser, GuinierParser

import io
import contextlib
from sys import version_info

logger = logging.getLogger(__name__)


class TestSasArgParser(unittest.TestCase):
    def minimal_guinier_parser_requires_file_argument(self):
        """
        Test that parser provides error if no file argument is provided.
        """
        basic_parser = GuinierParser("program", "description", "epilog")
        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stderr(output_catcher):
                _ = basic_parser.parse_args()
        except SystemExit:
            pass

        self.assertTrue(
            basic_parser.usage in output_catcher.getvalue(),
            msg="GuinierParser provides usage if no file provided",
        )
        self.assertTrue(
            "the following arguments are required: FILE"
            in output_catcher.getvalue(),
            msg="GuinierParser states that the FILE argument is missing if no file provided",
        )

    def minimal_parser_default_verbosity_level_is_0(self):
        """
        Test that the parser sets the verbosity to 0 if no args are provided
        """
        basic_parser = SASParser("program", "description", "epilog")
        parsed_arguments = basic_parser.parse_args()
        self.assertEqual(
            parsed_arguments.verbose,
            0,
            msg="SASParser default verbosity is 0",
        )

    def minimal_guinier_parser_default_verbosity_level_is_0(self):
        """
        Test that the Guinier parser sets the verbosity to 0 if no args are provided
        """
        basic_parser = GuinierParser("program", "description", "epilog")
        parsed_arguments = basic_parser.parse_args(["afile"])
        self.assertEqual(
            parsed_arguments.verbose,
            0,
            msg="GuinierParser default verbosity is 0",
        )

    def minimal_parser_accumulates_verbosity_level(self):
        """
        Test that the parser parser increases the verbosity level to two
        if -vv argument is provided.
        """
        basic_parser = SASParser("program", "description", "epilog")
        parsed_arguments = basic_parser.parse_args(["-vv"])
        self.assertEqual(
            parsed_arguments.verbose,
            2,
            msg="SASParser verbosity increased to 2 by -vv",
        )

    def minimal_guinier_parser_accumulates_verbosity_level(self):
        """
        Test that the parser parser increases the verbosity level to two
        if -vv argument is provided.
        """
        basic_parser = GuinierParser("program", "description", "epilog")
        parsed_arguments = basic_parser.parse_args(["afile", "-vv"])
        self.assertEqual(
            parsed_arguments.verbose,
            2,
            msg="GuinierParser verbosity increased to 2 by -vv",
        )

    def minimal_parser_provides_correct_version(self):
        """
        Test that parser provides the correct app version.
        """
        basic_parser = SASParser("program", "description", "epilog")
        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_catcher):
                _ = basic_parser.parse_args(["--version"])
        except SystemExit:
            pass

        self.assertTrue(
            freesas_version.version in output_catcher.getvalue(),
            msg="SASParser outputs consistent version",
        )
        self.assertTrue(
            freesas_version.date in output_catcher.getvalue(),
            msg="SASParser outputs consistent date",
        )

    def minimal_guinier_parser_provides_correct_version(self):
        """
        Test that parser provides the correct app version.
        """
        basic_parser = GuinierParser("program", "description", "epilog")
        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_catcher):
                _ = basic_parser.parse_args(["--version"])
        except SystemExit:
            pass

        self.assertTrue(
            freesas_version.version in output_catcher.getvalue(),
            msg="GuinierParser outputs consistent version",
        )
        self.assertTrue(
            freesas_version.date in output_catcher.getvalue(),
            msg="GuinierParser outputs consistent date",
        )


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        TestSasArgParser("minimal_parser_provides_correct_version")
    )
    test_suite.addTest(
        TestSasArgParser("minimal_guinier_parser_provides_correct_version")
    )
    test_suite.addTest(
        TestSasArgParser("minimal_parser_default_verbosity_level_is_0")
    )
    test_suite.addTest(
        TestSasArgParser("minimal_guinier_parser_default_verbosity_level_is_0")
    )
    test_suite.addTest(
        TestSasArgParser("minimal_parser_accumulates_verbosity_level")
    )
    test_suite.addTest(
        TestSasArgParser("minimal_guinier_parser_accumulates_verbosity_level")
    )
    test_suite.addTest(
        TestSasArgParser("minimal_guinier_parser_requires_file_argument")
    )
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())