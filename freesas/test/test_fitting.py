#!/usr/bin/python
# coding: utf-8

"""Test the functionality of fitting module"""

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "25/03/2022"


import unittest
import logging
import sys
import importlib
import platform
from io import StringIO
from freesas.fitting import set_logging_level


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

    @unittest.mock.patch.dict("sys.modules", {"nt": unittest.mock.MagicMock()})
    def test_get_linesep_returns_rn_if_output_is_stdout_on_windows(self):
        """
        Test that get_linesep() returns \r\n if output destination is sys.stdout on Windows.
        """
        # Reload to apply patches
        with unittest.mock.patch("sys.builtin_module_names", ["nt"]):
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

    @unittest.mock.patch.dict("sys.modules", {"nt": unittest.mock.MagicMock()})
    def test_get_linesep_returns_n_if_output_is_filestream_on_windows(self):
        """
        Test that get_linesep() returns \n if output destination is a filestream on Windows.
        """
        # Reload to apply patches
        with unittest.mock.patch("sys.builtin_module_names", ["nt"]):
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
