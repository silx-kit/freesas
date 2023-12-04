#!/usr/bin/python
# coding: utf-8

"""Test the functionality of SASParser and GuinierParser"""

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "25/03/2021"


import unittest
import logging
import io
import contextlib
from pathlib import Path
from .. import dated_version as freesas_version
from ..sas_argparser import SASParser, GuinierParser


logger = logging.getLogger(__name__)


class TestSasArgParser(unittest.TestCase):
    def test_minimal_guinier_parser_requires_file_argument(self):
        """
        Test that Guinier parser provides error if no file argument is provided.
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

    def test_minimal_guinier_parser_parses_list_of_files(self):
        """
        Test that the Guinier parsers parses a list of files.
        """
        basic_parser = GuinierParser("program", "description", "epilog")

        parsed_arguments = basic_parser.parse_args(["afile", "bfile", "cfile"])

        self.assertEqual(
            set(parsed_arguments.file),
            {"afile", "bfile", "cfile"},
            msg="GuinierParser parses list of files",
        )

    def test_add_file_argument_enables_SASParser_to_recognize_file_lists(
        self,
    ):
        """
        Test that add_file_argument adds the ability to parse a file list to SASParser.
        """
        basic_parser = SASParser("program", "description", "epilog")

        # Before running add_file_argument a file argument is not recognized
        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stderr(output_catcher):
                _ = basic_parser.parse_args(["afile"])
        except SystemExit:
            pass
        self.assertTrue(
            "unrecognized arguments: afile" in output_catcher.getvalue(),
            msg="Minimal SASParser does not recognize file argument",
        )

        basic_parser.add_file_argument(help_text="file help")
        parsed_arguments = basic_parser.parse_args(["afile", "bfile", "cfile"])

        self.assertEqual(
            set(parsed_arguments.file),
            {"afile", "bfile", "cfile"},
            msg="GuinierParser parses list of files",
        )

    def test_minimal_parser_usage_includes_program_name(self):
        """
        Test that minimal parser includes the provided program in the usage string.
        """
        basic_parser = SASParser("test❤️", "description", "epilog")

        self.assertTrue(
            "test❤️" in basic_parser.usage,
            msg="SASParser usage includes program name",
        )

    def test_minimal_guinier_parser_usage_includes_program_name(self):
        """
        Test that minimal parser includes the provided program in the usage string.
        """
        basic_parser = GuinierParser("test❤️", "description", "epilog")

        self.assertTrue(
            "test❤️" in basic_parser.usage,
            msg="GuinierParser usage includes program name",
        )

    def test_minimal_guinier_parser_help_includes_program_description_epilog(
        self,
    ):
        """
        Test that minimal guinier parser includes help includes
        the provided program name, description and epilog.
        """
        basic_parser = GuinierParser("test❤️", "description📚", "epilog🎦")
        output_catcher = io.StringIO()

        try:
            with contextlib.redirect_stdout(output_catcher):
                _ = basic_parser.parse_args(["--help"])
        except SystemExit:
            pass

        self.assertTrue(
            "test❤️" in output_catcher.getvalue(),
            msg="GuinierParser outputs program name in help",
        )

        self.assertTrue(
            "description📚" in output_catcher.getvalue(),
            msg="GuinierParser outputs description in help",
        )

        self.assertTrue(
            "epilog🎦" in output_catcher.getvalue(),
            msg="GuinierParser outputs eplilog name in help",
        )

    def test_minimal_parser_help_includes_program_description_epilog(self):
        """
        Test that minimal parser includes help includes
        the provided program name, description and epilog.
        """
        basic_parser = SASParser("test❤️", "description📚", "epilog🎦")
        output_catcher = io.StringIO()

        try:
            with contextlib.redirect_stdout(output_catcher):
                _ = basic_parser.parse_args(["--help"])
        except SystemExit:
            pass

        self.assertTrue(
            "test❤️" in output_catcher.getvalue(),
            msg="SASParser outputs program name in help",
        )

        self.assertTrue(
            "description📚" in output_catcher.getvalue(),
            msg="SASParser outputs description in help",
        )

        self.assertTrue(
            "epilog🎦" in output_catcher.getvalue(),
            msg="SASParser outputs eplilog name in help",
        )

    def test_minimal_parser_default_verbosity_level_is_0(self):
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

    def test_minimal_guinier_parser_default_verbosity_level_is_0(self):
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

    def test_minimal_parser_accumulates_verbosity_level(self):
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

    def test_minimal_guinier_parser_accumulates_verbosity_level(self):
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

    def test_minimal_parser_provides_correct_version(self):
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

    def test_minimal_guinier_parser_provides_correct_version(self):
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

    def test_minimal_guinier_parser_accepts_output_file_argument(self):
        """
        Test that minimal Guinier parser accepts one output file argument.
        """
        basic_parser = GuinierParser("program", "description", "epilog")
        parsed_arguments = basic_parser.parse_args(["afile", "-o", "out.file"])

        self.assertEqual(
            parsed_arguments.output,
            Path("out.file"),
            msg="Minimal GuinierParser accepts output file argument",
        )

    def test_add_output_filename_argument_adds_output_file_argument_to_SASParser(
        self,
    ):
        """
        Test that add_output_filename_argument adds one output file argument to as SASParser.
        """
        basic_parser = SASParser("program", "description", "epilog")

        # Before running add_output_filename_argument -o file is not regognized
        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stderr(output_catcher):
                _ = basic_parser.parse_args(["-o", "out.file"])
        except SystemExit:
            pass
        self.assertTrue(
            "unrecognized arguments: -o out.file" in output_catcher.getvalue(),
            msg="Minimal SASParser does not recognize -o argument",
        )

        basic_parser.add_output_filename_argument()
        parsed_arguments = basic_parser.parse_args(["-o", "out.file"])

        self.assertEqual(
            parsed_arguments.output,
            Path("out.file"),
            msg="SASParser accepts output file argument"
            "after running add_output_filename_argument()",
        )

    def test_minimal_guinier_parser_accepts_output_format_argument(self):
        """
        Test that minimal Guinier parser accepts one output data format argument.
        """
        basic_parser = GuinierParser("program", "description", "epilog")
        parsed_arguments = basic_parser.parse_args(["afile", "-f", "aformat"])

        self.assertEqual(
            parsed_arguments.format,
            "aformat",
            msg="Minimal GuinierParser accepts output data format argument",
        )

    def test_add_output_data_format_adds_output_format_argument_to_SASParser(
        self,
    ):
        """
        Test that add_output_data_format adds one output data format argument to as SASParser.
        """
        basic_parser = SASParser("program", "description", "epilog")

        # Before running add_output_filename_argument -o file is not regognized
        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stderr(output_catcher):
                _ = basic_parser.parse_args(["-f", "aformat"])
        except SystemExit:
            pass
        self.assertTrue(
            "unrecognized arguments: -f aformat" in output_catcher.getvalue(),
            msg="Minimal SASParser does not recognize -f argument",
        )

        basic_parser.add_output_data_format()
        parsed_arguments = basic_parser.parse_args(["-f", "aformat"])

        self.assertEqual(
            parsed_arguments.format,
            "aformat",
            msg="SASParser accepts output data format argument"
            "after running add_output_data_format()",
        )

    def test_minimal_guinier_parser_accepts_q_unit_argument(self):
        """
        Test that minimal Guinier parser accepts a q unit argument.
        """
        basic_parser = GuinierParser("program", "description", "epilog")
        parsed_arguments = basic_parser.parse_args(["afile", "-u", "nm"])

        self.assertEqual(
            parsed_arguments.unit,
            "nm",
            msg="Minimal GuinierParser accepts q unit argument",
        )

    def test_add_q_unit_argument_adds_add_q_unit_argument_to_SASParser(
        self,
    ):
        """
        Test that add_q_unit_argument adds a q unit argument to as SASParser.
        """
        basic_parser = SASParser("program", "description", "epilog")

        # Before running add_output_filename_argument -o file is not regognized
        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stderr(output_catcher):
                _ = basic_parser.parse_args(["-u", "nm"])
        except SystemExit:
            pass
        self.assertTrue(
            "unrecognized arguments: -u nm" in output_catcher.getvalue(),
            msg="Minimal SASParser does not recognize -u argument",
        )

        basic_parser.add_q_unit_argument()
        parsed_arguments = basic_parser.parse_args(["-u", "nm"])

        self.assertEqual(
            parsed_arguments.unit,
            "nm",
            msg="SASParser accepts q unit argument after running add_q_unit_argument()",
        )

    def test_SASParser_q_unit_argument_allows_predefined_units(
        self,
    ):
        """
        Test that the q unit argument of a SASparser accepts "nm", "Å", "A".
        """
        basic_parser = SASParser("program", "description", "epilog")
        basic_parser.add_q_unit_argument()

        parsed_arguments = basic_parser.parse_args(["-u", "nm"])
        self.assertEqual(
            parsed_arguments.unit,
            "nm",
            msg="SASParser accepts unit format nm",
        )

        parsed_arguments = basic_parser.parse_args(["-u", "A"])
        self.assertEqual(
            parsed_arguments.unit,
            "Å",
            msg="SASParser accepts unit format A",
        )

        parsed_arguments = basic_parser.parse_args(["-u", "Å"])
        self.assertEqual(
            parsed_arguments.unit,
            "Å",
            msg="SASParser accepts unit format A",
        )

    def test_SASParser_q_unit_argument_does_not_allow_not_predefined_units(
        self,
    ):
        """
        Test that the q unit argument of a SASparser does not accept a
        unit that is not "nm", "Å", "A".
        """
        basic_parser = SASParser("program", "description", "epilog")
        basic_parser.add_q_unit_argument()

        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stderr(output_catcher):
                _ = basic_parser.parse_args(["-u", "m"])
        except SystemExit:
            pass
        self.assertTrue(
            "argument -u/--unit: invalid choice: 'm' (choose from 'nm', 'Å', 'A')"
            in output_catcher.getvalue(),
            msg="SASParser does not accept '-u m' argument",
        )

    def test_SASParser_q_unit_A_gets_converted_to_Å(
        self,
    ):
        """
        Test that for a SASParder q unit input "A" gets converted to "Å".
        """
        basic_parser = SASParser("program", "description", "epilog")
        basic_parser.add_q_unit_argument()

        parsed_arguments = basic_parser.parse_args(["-u", "A"])
        self.assertEqual(
            parsed_arguments.unit,
            "Å",
            msg="SASParser converts unit input 'A' to 'Å'",
        )

    def test_GuinierParser_q_unit_argument_allows_predefined_units(
        self,
    ):
        """
        Test that the q unit argument of a Guinierparser accepts "nm", "Å", "A".
        """
        basic_parser = GuinierParser("program", "description", "epilog")

        parsed_arguments = basic_parser.parse_args(["afile", "-u", "nm"])
        self.assertEqual(
            parsed_arguments.unit,
            "nm",
            msg="SASParser accepts unit format nm",
        )

        parsed_arguments = basic_parser.parse_args(["afile", "-u", "A"])
        self.assertEqual(
            parsed_arguments.unit,
            "Å",
            msg="SASParser accepts unit format A",
        )

        parsed_arguments = basic_parser.parse_args(["afile", "-u", "Å"])
        self.assertEqual(
            parsed_arguments.unit,
            "Å",
            msg="SASParser accepts unit format A",
        )

    def test_GuinierParser_q_unit_argument_does_not_allow_not_predefined_units(
        self,
    ):
        """
        Test that the q unit argument of a Guinierparser does not accept a
        unit that is not "nm", "Å", "A".
        """
        basic_parser = GuinierParser("program", "description", "epilog")

        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stderr(output_catcher):
                _ = basic_parser.parse_args(["afile", "-u", "m"])
        except SystemExit:
            pass
        self.assertTrue(
            "argument -u/--unit: invalid choice: 'm' (choose from 'nm', 'Å', 'A')"
            in output_catcher.getvalue(),
            msg="SASParser does not accept '-u m' argument",
        )

    def test_GuinierParser_q_unit_A_gets_converted_to_Å(
        self,
    ):
        """
        Test that for a GuinierParser q unit input "A" gets converted to "Å".
        """
        basic_parser = GuinierParser("program", "description", "epilog")

        parsed_arguments = basic_parser.parse_args(["afile", "-u", "A"])
        self.assertEqual(
            parsed_arguments.unit,
            "Å",
            msg="SASParser converts unit input 'A' to 'Å'",
        )

    def test_add_argument_adds_an_argument_to_a_SASParser(
        self,
    ):
        """
        Test that new arguments can be added to SASParser.
        """
        basic_parser = SASParser("program", "description", "epilog")

        # Before running add_argument -c
        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stderr(output_catcher):
                _ = basic_parser.parse_args(["-c"])
        except SystemExit:
            pass
        self.assertTrue(
            "unrecognized arguments: -c" in output_catcher.getvalue(),
            msg="Minimal SASParser does not recognize -c argument",
        )

        basic_parser.add_argument(
            "-c",
            "--check",
            action="store_true",
        )

        parsed_arguments = basic_parser.parse_args(["-c"])
        self.assertEqual(
            parsed_arguments.check,
            True,
            msg="-c argument added to SASParser",
        )

    def test_add_argument_adds_an_argument_to_a_GuinierParser(
        self,
    ):
        """
        Test that new arguments can be added to GuinierParser.
        """
        basic_parser = GuinierParser("program", "description", "epilog")

        # Before running add_argument -c
        output_catcher = io.StringIO()
        try:
            with contextlib.redirect_stderr(output_catcher):
                _ = basic_parser.parse_args(["afile", "-c"])
        except SystemExit:
            pass
        print(output_catcher.getvalue())
        self.assertTrue(
            "unrecognized arguments: -c" in output_catcher.getvalue(),
            msg="Minimal GuinierParser does not recognize -c argument",
        )

        basic_parser.add_argument(
            "-c",
            "--check",
            action="store_true",
        )

        parsed_arguments = basic_parser.parse_args(["afile", "-c"])
        self.assertEqual(
            parsed_arguments.check,
            True,
            msg="-c argument added to GuinierParser",
        )


def suite():
    """Build a test suite from the TestSasArgParser class"""
    test_suite = unittest.TestSuite()
    for class_element in dir(TestSasArgParser):
        if class_element.startswith("test"):
            test_suite.addTest(TestSasArgParser(class_element))
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
