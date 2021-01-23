"""End to end tests for auto_gpa.py """

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "27/12/2020"

import unittest
import pathlib
import logging
from subprocess import run, PIPE, STDOUT
from os import linesep

logger = logging.getLogger(__name__)


class TestFreeSAS(unittest.TestCase):

    test_location = pathlib.Path(__file__)
    test_data_location = pathlib.Path(test_location.parent, "e2etest_data")
    bsa_filename = pathlib.Path(test_data_location, "bsa_005_sub.dat")
    sas_curve2_filename = pathlib.Path(test_data_location, "SASDF52.dat")

    def test_one_bm29_bsa_without_arguments(self):
        """
        Test whether auto_gpa.py on BSA data from BM29 returns one line.
        """
        run_auto_gpa = run(
            ["auto_gpa.py", str(self.bsa_filename)],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_auto_gpa.returncode, 0, msg="auto_gpa.py completed well"
        )
        run_auto_gpa_output = str(run_auto_gpa.stdout, "utf-8")[:-1]
        self.assertFalse(
            linesep in run_auto_gpa_output, "More than one line of output"
        )
        self.assertTrue(
            str(self.bsa_filename) in run_auto_gpa_output,
            msg="filename of testdata not found in output",
        )
        self.assertTrue("I₀" in run_auto_gpa_output, msg="I₀ missing?")
        self.assertTrue("" in run_auto_gpa_output, msg="Rg missing?")

    def test_two_easy_files_without_arguments(self):
        """
        Test whether auto_gpa.py with two datasets returns two lines.
        """
        run_auto_gpa = run(
            [
                "auto_gpa.py",
                str(self.bsa_filename),
                str(self.sas_curve2_filename),
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_auto_gpa.returncode,
            0,
            msg="auto_gpa.py for 2 files completed well",
        )
        self.assertTrue(
            str(self.bsa_filename) in str(run_auto_gpa.stdout, "utf-8")
            and str(self.sas_curve2_filename)
            in str(run_auto_gpa.stdout, "utf-8"),
            "filename of testdata not found in output",
        )
        run_auto_gpa_output = str(run_auto_gpa.stdout, "utf-8")[:-1].split(
            linesep
        )
        print(run_auto_gpa_output)
        self.assertEqual(
            len(run_auto_gpa_output), 2, msg="Number of output lines is not 2"
        )
        for result in run_auto_gpa_output:
            self.assertTrue(
                str(self.bsa_filename) in result
                or str(self.sas_curve2_filename) in result,
                "filename of testdata not found in result line",
            )
            self.assertTrue("I" in result, "I₀ missing?")
            self.assertTrue("" in result, "Rg missing?")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestFreeSAS("test_one_BM29_bsa_without_arguments"))
    test_suite.addTest(TestFreeSAS("test_two_easy_files_without_arguments"))
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
