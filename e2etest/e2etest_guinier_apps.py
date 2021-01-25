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
    cwd = pathlib.Path.cwd()
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
            "One line of output of {app_name}",
        )
        self.assertTrue(
            str(self.bsa_filename) in run_app_output,
            msg="filename of testdata found in output of {app_name}",
        )
        self.assertTrue(
            "I₀" in run_app_output, msg=f"I₀ in {app_name}"
        )
        self.assertTrue(
            "Rg" in run_app_output, msg=f"Rg in {app_name}"
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
            f"filename of testdata found in output in output of {app_name}",
        )
        run_app_output = str(run_app.stdout, "utf-8")[:-1].split(linesep)
        self.assertEqual(
            len(run_app_output),
            2,
            msg=f"Number of output lines is 2 in output of {app_name}",
        )
        for result in run_app_output:
            self.assertTrue(
                str(self.bsa_filename) in result
                or str(self.sas_curve2_filename) in result,
                f"filename of testdata not found in result line for {app_name}",
            )
            self.assertTrue("I" in result, f"I₀ in {app_name} output")
            self.assertTrue("Rg" in result, f"Rg in {app_name} output")

    def test_csv_output_with_two_files(self):
        app_name: str = self.extra_arg["app"]
        test_output_name = pathlib.Path(self.cwd, f"{app_name}.csv")
        try:
            test_output_name.unlink()
        except FileNotFoundError:
            pass
        run_app = run(
            [
                app_name,
                str(self.bsa_filename),
                str(self.sas_curve2_filename),
                "-o",
                test_output_name,
                "-f",
                "csv",
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode,
            0,
            msg=f"{app_name} for 2 files to csv completed well",
        )
        self.assertEqual(
            run_app.stdout,
            b'',
            msg=f"{app_name} for 2 files to csv provided no output to stdout",
        )
        with open(test_output_name, 'r') as test_output_file:
            test_output_header = test_output_file.readline()[:-1].split(",")
            test_output_result = test_output_file.readlines()
        test_output_result = [line[:-1].split(",") for line in test_output_result]
        expected_header = {'File', 'Rg', 'Rg StDev', 'I(0)', 'I(0) StDev', 'First point', 'Last point', 'Quality', 'Aggregated'}
        self.assertEqual(expected_header^set(test_output_header), set(), msg=f"header created by {app_name} contains the expected element")
        self.assertEqual(len(test_output_header), len(set(test_output_header)), msg=f"header created by {app_name} contains no duplicates")

        file_name_index = test_output_header.index('File')
        Rg_index = test_output_header.index('Rg')
        Rg_std_index = test_output_header.index('Rg StDev')
        I0_index = test_output_header.index('I(0)')
        I0_std_index = test_output_header.index('I(0) StDev')
        first_point_index = test_output_header.index('First point')
        last_point_index = test_output_header.index('Last point')
        quality_index = test_output_header.index('Quality')
        aggregated_index = test_output_header.index('Aggregated')

        self.assertEqual(len(test_output_result), 2, msg=f"result file created by {app_name} contains 2 results")
        self.assertEqual({str(self.bsa_filename),str(self.sas_curve2_filename)}, {test_output_result[0][file_name_index], test_output_result[1][file_name_index]}, msg=f"result file created by {app_name} contains results for the right files" )
        for result in  test_output_result:
            if result[file_name_index] == str(self.bsa_filename):
                self.assertTrue(abs(float(result[Rg_index]) - 2.95) <= 0.1, msg=f"Rg for BSA by {app_name} between 2.85 and 3.05")
                self.assertTrue(float(result[Rg_std_index])  < 0.3, msg=f"Rg std. for BSA by {app_name} below 0.3")
                self.assertTrue(abs(float(result[I0_index]) - 62.5) <= 2.5, msg=f"I0 for BSA by {app_name} between 60 and 65")
                self.assertTrue(float(result[I0_std_index])  < 0.25, msg=f"I0 std. for BSA by {app_name} below 0.25")
                self.assertTrue(float(result[first_point_index]) < 35, msg=f"Guinier region start for BSA by {app_name} below 35")
                self.assertTrue(float(result[last_point_index])  > 85 , msg=f"Guinier region end for BSA by {app_name} above 85")
                self.assertTrue(float(result[quality_index]) > 0.8, msg=f"Fit quality BSA by {app_name} above 0.8")
                self.assertTrue(float(result[aggregated_index]) < 0.05, msg=f"Aggregation number for BSA by {app_name} below 0.05")
            elif result[file_name_index] == str(self.sas_curve2_filename):
                self.assertTrue(abs(float(result[Rg_index]) - 2.95) <= 0.1, msg=f"Rg for SASDF52 by {app_name} between 2.85 and 3.05")
                self.assertTrue(float(result[Rg_std_index])  < 0.015, msg=f"Rg std. for SASDF52 by {app_name} below 0.015")
                self.assertTrue(abs(float(result[I0_index]) - 61) <= 1, msg=f"I0 for SASDF52 by {app_name} between 60 and 62")
                self.assertTrue(float(result[I0_std_index])  < 0.1, msg=f"I0 std. for SASDF52 by {app_name} below 0.1")
                self.assertTrue(float(result[first_point_index])  < 35, msg=f"Guinier region start for SASDF52 by {app_name} below 35")
                self.assertTrue(float(result[last_point_index]) > 85, msg=f"Guinier region end for SASDF52 by {app_name} above 85")
                self.assertTrue(float(result[quality_index]) > 0.8, msg=f"Fit quality SASDF52 by {app_name} above 0.8")
                self.assertTrue(float(result[aggregated_index] ) < 0.1, msg=f"Aggregation number for SASDF52 by {app_name} below 0.1")


def suite():
    test_suite = unittest.TestSuite()
    for app in ["auto_gpa.py", "auto_guinier.py", "autorg.py"]:
        test_suite.addTest(
            TestFreeSAS("test_one_bm29_bsa_without_arguments", app=app)
        )
        test_suite.addTest(
            TestFreeSAS("test_two_easy_files_without_arguments", app=app)
        )
        test_suite.addTest(
            TestFreeSAS("test_csv_output_with_two_files", app=app)
        )
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
