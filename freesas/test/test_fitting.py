#!/usr/bin/python
# coding: utf-8

"""Test the functionality of fitting module"""

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "02/04/2021"


import unittest
from unittest.mock import patch
import logging
import sys
import importlib
import platform
from io import StringIO, TextIOWrapper
from pathlib import Path
from freesas.fitting import (
    set_logging_level,
    get_output_destination,
    get_header,
)

if sys.version_info.minor > 6:
    from unittest.mock import mock_open
else:
    from .mock_open_38 import mock_open


logger = logging.getLogger(__name__)


class TestFitting(unittest.TestCase):
    def reload_os_and_fitting(self):
        """Some tests patch os and need to reload the modules"""
        os = importlib.import_module("os")
        os = importlib.reload(os)
        fit = importlib.import_module("freesas.fitting")
        fit = importlib.reload(fit)
        return fit

    def test_set_logging_level_does_not_change_logging_level_if_input_lower_1(
        self,
    ):
        """
        Test that the logging level only gets changed if the requested level is > 0.
        """
        initial_logging_level = logging.root.level
        set_logging_level(0)
        self.assertEqual(
            logging.root.level,
            initial_logging_level,
            msg="settting verbosity to 0 dos not affect logging level",
        )
        set_logging_level(-2)
        self.assertEqual(
            logging.root.level,
            initial_logging_level,
            msg="settting verbosity to -2 does not affect logging level",
        )
        # Ensure that initial logging level is restored
        logging.root.setLevel(initial_logging_level)

    def test_set_logging_level_sets_logging_to_INFO_if_input_is_1(
        self,
    ):
        """
        Test that the logging level gets changed to INFO if verbosity is 1.
        """
        initial_logging_level = logging.root.level
        # Ensure that the function actually changes the level
        logging.root.setLevel(logging.WARNING)

        set_logging_level(1)
        self.assertEqual(
            logging.root.level,
            logging.INFO,
            msg="settting verbosity to 1 sets logging level to INFO",
        )

        # Ensure that initial logging level is restored
        logging.root.setLevel(initial_logging_level)

    def test_set_logging_level_sets_logging_to_DEBUG_if_input_is_2_or_more(
        self,
    ):
        """
        Test that the logging level gets changed to DEBUG if verbosity is 2 or larger.
        """
        initial_logging_level = logging.root.level
        # Ensure that the function actually changes the level
        logging.root.setLevel(logging.WARNING)

        set_logging_level(2)
        self.assertEqual(
            logging.root.level,
            logging.DEBUG,
            msg="settting verbosity to 2 sets logging level to DEBUG",
        )

        set_logging_level(3)
        self.assertEqual(
            logging.root.level,
            logging.DEBUG,
            msg="settting verbosity to 3 sets logging level to DEBUG",
        )

        # Ensure that initial logging level is restored
        logging.root.setLevel(initial_logging_level)

    @patch.dict("sys.modules", {"nt": unittest.mock.MagicMock()})
    def test_get_linesep_returns_rn_if_output_is_stdout_on_windows(self):
        """
        Test that get_linesep() returns \r\n if output destination is sys.stdout on Windows.
        """
        # Reload to apply patches
        with patch("sys.builtin_module_names", ["nt"]):
            fit = self.reload_os_and_fitting()

        get_linesep = getattr(fit, "get_linesep")
        self.assertEqual(get_linesep(sys.stdout), "\r\n")

        # Cleanup
        _ = self.reload_os_and_fitting()

    def test_get_linesep_returns_n_if_output_is_stdout_on_posix(
        self,
    ):
        """
        Test that get_linesep() returns \n if output destination is sys.stdout on Posix.
        Only should run on posix.
        """

        fit = importlib.import_module("freesas.fitting")
        get_linesep = getattr(fit, "get_linesep")

        self.assertEqual(get_linesep(sys.stdout), "\n")

    @patch.dict("sys.modules", {"nt": unittest.mock.MagicMock()})
    def test_get_linesep_returns_n_if_output_is_filestream_on_windows(self):
        """
        Test that get_linesep() returns \n if output destination is a filestream on Windows.
        """
        # Reload to apply patches
        with patch("sys.builtin_module_names", ["nt"]):
            fit = self.reload_os_and_fitting()

        get_linesep = getattr(fit, "get_linesep")

        output_dest = StringIO()
        self.assertEqual(get_linesep(output_dest), "\n")

        # Cleanup
        _ = self.reload_os_and_fitting()

    def test_get_linesep_returns_n_if_output_is_filestream_on_posix(
        self,
    ):
        """
        Test that get_linesep() returns \n if output destination is filestream on Posix.
        Only should run on posix.
        """

        fit = importlib.import_module("freesas.fitting")
        get_linesep = getattr(fit, "get_linesep")

        output_dest = StringIO()
        self.assertEqual(get_linesep(output_dest), "\n")

    @patch("__main__.open", mock_open())
    def test_get_output_destination_with_path_input_returns_writable_testIO(
        self,
    ):
        """Test that by calling get_output_destination with a Path as input
        we obtain write access to the file of Path"""
        with get_output_destination(Path("test")) as destination:
            self.assertEqual(
                type(destination),
                TextIOWrapper,
                msg="file destination has type TextIOWrapper",
            )
            self.assertTrue(
                destination.writable(),
                msg="file destination is writable",
            )

    def test_get_output_destination_withou_input_returns_stdout(
        self,
    ):
        """Test that by calling get_output_destination without input
        we obtain sys.stdout"""
        destination = get_output_destination()
        self.assertEqual(
            destination,
            sys.stdout,
            msg="default destination is sys.stdout",
        )

    def test_get_header_for_csv(
        self,
    ):
        """Test that by calling get_header with input csv we get the correct line"""
        header = get_header("csv", "linesep")
        self.assertEqual(
            header,
            "File,Rg,Rg StDev,I(0),I(0) StDev,First point,Last point,Quality,Aggregatedlinesep",
            msg="csv header is correct",
        )

    def test_get_header_for_ssv(
        self,
    ):
        """Test that by calling get_header with input ssv we get an empty string"""
        header = get_header("ssv", "linesep")
        self.assertEqual(
            header,
            "",
            msg="ssv header is correct",
        )

    def test_get_header_for_native(
        self,
    ):
        """Test that by calling get_header with input native we get an empty string"""
        header = get_header("native", "linesep")
        self.assertEqual(
            header,
            "",
            msg="native header is correct",
        )

    def test_get_header_without_input_format(
        self,
    ):
        """Test that by calling get_header without input format we get an empty string"""
        header = get_header(None, "linesep")
        self.assertEqual(
            header,
            "",
            msg="header for undefined format is correct",
        )


def suite():
    """Build a test suite from the TestFitting class"""
    test_suite = unittest.TestSuite()
    for class_element in dir(TestFitting):
        if platform.system() == "Windows":
            if (
                class_element.startswith("test")
                and not "posix" in class_element
            ):
                test_suite.addTest(TestFitting(class_element))
        else:
            if class_element.startswith("test"):
                test_suite.addTest(TestFitting(class_element))
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
