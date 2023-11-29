"""End to end tests for auto_gpa.py """

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "29/11/2023"

import unittest
import pathlib
import logging
from platform import system
from subprocess import run, PIPE, STDOUT
from os import linesep
from os.path import normpath
from freesas.test.utilstest import get_datafile

logger = logging.getLogger(__name__)

if system() == "Windows":
    cormapy = "cormapy.exe"
else:
    cormapy = "cormapy"


class TestCormap(unittest.TestCase):
    """End to end tests for free_bift"""

    cwd = pathlib.Path.cwd()
    bsa_filename = pathlib.Path(get_datafile("bsa_005_sub.dat"))

    def test_1_input_file_results_no_correlation(self):
        """
        Test that cormap does not list a correlation if only one file is provided.
        """

        run_app = run(
            [cormapy, normpath(str(self.bsa_filename))],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )

        if system() == "Windows":
            run_app_output_raw = str(run_app.stdout)[:-1].replace("\\\\", "\\")
            run_app_output = [
                line.replace("\\n", "").replace("\\r", "")
                for line in run_app_output_raw.split("\\n")[:-1]
            ]
        else:
            run_app_output_raw = str(run_app.stdout, "utf-8")[:-1]
            run_app_output = run_app_output_raw.split(linesep)[:-1]

        self.assertEqual(
            run_app.returncode, 0, msg="cormapy on BM29 BSA completed well"
        )

        corr_table_head_pos = [
            line_number
            for line_number, line in enumerate(run_app_output)
            if "Pr(>C)" in line
        ][0]

        self.assertEqual(
            run_app_output[corr_table_head_pos + 1],
            "",
            msg="Correlation table is empty",
        )

    def test_correlation_of_file_with_itself_is_1(self):
        """
        Test that the auto-correlation of a file is 1.
        """

        run_app = run(
            [
                cormapy,
                normpath(str(self.bsa_filename)),
                normpath(str(self.bsa_filename)),
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )

        if system() == "Windows":
            run_app_output_raw = str(run_app.stdout)[:-1].replace("\\\\", "\\")
            run_app_output = [
                line.replace("\\n", "").replace("\\r", "")
                for line in run_app_output_raw.split("\\n")[:-1]
            ]
        else:
            run_app_output_raw = str(run_app.stdout, "utf-8")[:-1]
            run_app_output = run_app_output_raw.split(linesep)[:-1]

        self.assertEqual(
            run_app.returncode,
            0,
            msg="auto-correlation cormapy on BM29 BSA completed well",
        )

        corr_table_head_pos = [
            line_number
            for line_number, line in enumerate(run_app_output)
            if "Pr(>C)" in line
        ][0]

        self.assertEqual(
            run_app_output[corr_table_head_pos + 1].split(),
            [
                "1",
                "vs.",
                "2",
                "0",
                "1.000000",
            ],
            msg="Auto-correaltion is 1",
        )


def suite():
    """Build test suite for free_bift"""
    test_suite = unittest.TestSuite()

    test_suite.addTest(TestCormap("test_1_input_file_results_no_correlation"))
    test_suite.addTest(TestCormap("test_correlation_of_file_with_itself_is_1"))

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
