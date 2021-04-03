#!/usr/bin/python
# coding: utf-8

"""Test the functionality of fitting module"""

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "02/04/2021"


import unittest
from unittest.mock import patch, MagicMock
import logging
import sys
import importlib
import platform
from io import StringIO, TextIOWrapper
import pathlib
from freesas.fitting import (
    set_logging_level,
    get_output_destination,
    get_header,
    collect_files,
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
            msg="setting verbosity to 0 dos not affect logging level",
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

    @patch.dict("sys.modules", {"nt": MagicMock()})
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
        self.reload_os_and_fitting()

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

    @patch.dict("sys.modules", {"nt": MagicMock()})
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

    def test_get_output_destination_with_path_input_returns_writable_IO(
        self,
    ):
        """Test that by calling get_output_destination with a Path as input
        we obtain write access to the file of Path"""
        mocked_open = mock_open()
        with patch("builtins.open", mocked_open):
            with get_output_destination(pathlib.Path("test")) as destination:
                self.assertTrue(
                    destination.writable(),
                    msg="file destination is writable",
                )
        mocked_open.assert_called_once_with(pathlib.Path("test"), "w")

    def test_get_output_destination_without_input_returns_stdout(
        self,
    ):
        """Test that by calling get_output_destination without input
        we obtain sys.stdout"""
        with get_output_destination() as destination:
            self.assertEqual(
                destination,
                sys.stdout,
                msg="default destination is sys.stdout",
            )

    def test_closing_get_output_destination_does_not_close_stdout(
        self,
    ):
        """Test that get_output_destination() can be safely used without closing sys.stdout"""
        with get_output_destination() as _:
            pass
        sys.stdout.write("test")

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

    def test_collect_files_only_returns_existing_files(self):
        """Test that collect_files discards strings that do not match an existing file"""

        def os_stat_mock(path):
            if "good" in path.name:
                pass
            else:
                raise ValueError

        mocked_stat = MagicMock(side_effect=os_stat_mock)
        with patch("os.stat", mocked_stat):
            local_pathlib = importlib.import_module("pathlib")
            local_pathlib = importlib.reload(local_pathlib)
            MyPath = getattr(local_pathlib, "Path")
            fit = importlib.import_module("freesas.fitting")
            fit = importlib.reload(fit)
            collect_files = getattr(fit, "collect_files")
            self.assertEqual(
                collect_files(["testgood", "testbad"]),
                [MyPath("testgood")],
            )
        # Reload without the patch
        local_pathlib = importlib.reload(local_pathlib)
        self.reload_os_and_fitting()

    @patch("platform.system", MagicMock(return_value="Windows"))
    def test_collect_files_globs_on_windows(self):
        """Test that collect_files globs on Windows if no existent files provided"""

        def os_stat_mock(path):
            raise ValueError

        mocked_stat = MagicMock(side_effect=os_stat_mock)
        mocked_glob = MagicMock(
            side_effect=[
                (p for p in [pathlib.Path("pathA"), pathlib.Path("pathB")])
            ]
        )
        with patch("os.stat", mocked_stat):
            with patch.object(pathlib.Path, "glob", mocked_glob):
                fit = importlib.import_module("freesas.fitting")
                fit = importlib.reload(fit)
                collect_files = getattr(fit, "collect_files")
                self.assertEqual(
                    collect_files(["testgood"]),
                    [pathlib.Path("pathA"), pathlib.Path("pathB")],
                )
        mocked_glob.assert_called_once()

        # Reload without the patch
        self.reload_os_and_fitting()


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
