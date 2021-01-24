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

    def __init__(self, testName, **extra_kwargs):
        super(TestFreeSAS, self).__init__(testName)
        self.extra_arg = extra_kwargs

    def test_one_bm29_bsa_without_arguments(self):
        """
        Test whether auto_gpa.py on BSA data from BM29 returns one line.
        """
        app_name: str = self.extra_arg["app"]
        run_app = run(
            [app_name, str(self.bsa_filename)],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode, 0, msg=f"{app_name} completed well"
        )
        run_app_output = str(run_app.stdout, "utf-8")[:-1]
        self.assertFalse(
            linesep in run_app_output,
            "More than one line of output of {app_name}",
        )
        self.assertTrue(
            str(self.bsa_filename) in run_app_output,
            msg="filename of testdata not found in output of {app_name}",
        )
        self.assertTrue(
            "I₀" in run_app_output, msg=f"I₀ missing? in {app_name}"
        )
        self.assertTrue(
            "Rg" in run_app_output, msg=f"Rg missing? in {app_name}"
        )

    def test_two_easy_files_without_arguments(self):
        """
        Test whether auto_gpa.py with two datasets returns two lines.
        """
        app_name: str = self.extra_arg["app"]
        run_app = run(
            [
                app_name,
                str(self.bsa_filename),
                str(self.sas_curve2_filename),
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode,
            0,
            msg=f"{app_name} for 2 files completed well",
        )
        self.assertTrue(
            str(self.bsa_filename) in str(run_app.stdout, "utf-8")
            and str(self.sas_curve2_filename) in str(run_app.stdout, "utf-8"),
            f"filename of testdata not found in output in output of {app_name}",
        )
        run_app_output = str(run_app.stdout, "utf-8")[:-1].split(linesep)
        self.assertEqual(
            len(run_app_output),
            2,
            msg=f"Number of output lines is not 2 in output of {app_name}",
        )
        for result in run_app_output:
            self.assertTrue(
                str(self.bsa_filename) in result
                or str(self.sas_curve2_filename) in result,
                f"filename of testdata not found in result line for {app_name}",
            )
            self.assertTrue("I" in result, f"I₀ missing? in {app_name}")
            self.assertTrue("Rg" in result, f"Rg missing? in {app_name}")


def suite():
    test_suite = unittest.TestSuite()
    for app in ["auto_gpa.py", "auto_guinier.py", "autorg.py"]:
        test_suite.addTest(
            TestFreeSAS("test_one_bm29_bsa_without_arguments", app=app)
        )
        test_suite.addTest(
            TestFreeSAS("test_two_easy_files_without_arguments", app=app)
        )
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
