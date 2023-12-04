"""End to end tests for auto_gpa.py """

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "29/11/2023"

import unittest
import pathlib
import logging
from subprocess import run, PIPE, STDOUT
from os import linesep
from os.path import normpath
from platform import system
from freesas.test.utilstest import get_datafile
logger = logging.getLogger(__name__)


class TestFreeSAS(unittest.TestCase):
    cwd = pathlib.Path.cwd()
    test_location = pathlib.Path(__file__)
    test_data_location = pathlib.Path(test_location.parent, "e2etest_data")
    bsa_filename = pathlib.Path(get_datafile("bsa_005_sub.dat"))
    sas_curve2_filename = pathlib.Path(get_datafile( "SASDF52.dat"))
    SASDFX7 = pathlib.Path(get_datafile("SASDFX7.dat"))

    def __init__(self, testName, **extra_kwargs):
        super().__init__(testName)
        self.extra_arg = extra_kwargs

    def test_one_bm29_bsa_without_arguments(self):
        """
        Test whether guinier fit apps on BSA data from BM29 returns one line.
        """
        app_name: str = self.extra_arg["app"]
        run_app = run(
            [app_name, normpath(str(self.bsa_filename))],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode, 0, msg=f"{app_name} completed well"
        )
        if system() == "Windows":
            run_app_output = str(run_app.stdout)[:-1].replace("\\\\", "\\")
        else:
            run_app_output = str(run_app.stdout, "utf-8")[:-1]
        self.assertFalse(
            linesep in run_app_output,
            "One line of output of {app_name}",
        )
        self.assertTrue(
            str(self.bsa_filename) in run_app_output,
            msg="filename of testdata found in output of {app_name}",
        )
        self.assertTrue("I0" in run_app_output, msg=f"I0 in {app_name}")
        self.assertTrue("Rg" in run_app_output, msg=f"Rg in {app_name}")

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
        if system() == "Windows":
            run_app_output_raw = str(run_app.stdout)[:-1].replace("\\\\", "\\")
            run_app_output = run_app_output_raw.split("\\n")[:-1]
        else:
            run_app_output_raw = str(run_app.stdout, "utf-8")[:-1]
            run_app_output = run_app_output_raw.split(linesep)
        self.assertTrue(
            str(self.bsa_filename) in run_app_output_raw
            and str(self.sas_curve2_filename) in run_app_output_raw,
            f"filename of testdata found in output in output of {app_name}",
        )

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
            self.assertTrue("I" in result, f"I0 in {app_name} output")
            self.assertTrue("Rg" in result, f"Rg in {app_name} output")

    def test_csv_output_with_two_files(self):
        """Test whether the output in csv format is consistent"""
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
                str(test_output_name),
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
            b"",
            msg=f"{app_name} for 2 files to csv provided no output to stdout",
        )
        with open(str(test_output_name), "r") as test_output_file:
            test_output_header = test_output_file.readline()[:-1].split(",")
            test_output_result = test_output_file.readlines()
        test_output_result = [
            line[:-1].split(",") for line in test_output_result
        ]
        expected_header = {
            "File",
            "Rg",
            "Rg StDev",
            "I(0)",
            "I(0) StDev",
            "First point",
            "Last point",
            "Quality",
            "Aggregated",
        }
        self.assertEqual(
            expected_header ^ set(test_output_header),
            set(),
            msg=f"header created by {app_name} contains the expected element",
        )
        self.assertEqual(
            len(test_output_header),
            len(set(test_output_header)),
            msg=f"header created by {app_name} contains no duplicates",
        )
        index = {}
        for header_item in expected_header:
            index[header_item] = test_output_header.index(header_item)

        self.assertEqual(
            len(test_output_result),
            2,
            msg=f"result file created by {app_name} contains 2 results",
        )
        self.assertEqual(
            {str(self.bsa_filename), str(self.sas_curve2_filename)},
            {
                test_output_result[0][index["File"]],
                test_output_result[1][index["File"]],
            },
            msg=f"result file created by {app_name} contains results for the right files",
        )
        for result in test_output_result:
            if result[index["File"]] == str(self.bsa_filename):
                self.assertTrue(
                    abs(float(result[index["Rg"]]) - 2.95) <= 0.1,
                    msg=f"Rg for BSA by {app_name} between 2.85 and 3.05",
                )
                self.assertTrue(
                    float(result[index["Rg StDev"]]) < 0.3,
                    msg=f"Rg std. for BSA by {app_name} below 0.3",
                )
                self.assertTrue(
                    abs(float(result[index["I(0)"]]) - 62.5) <= 2.5,
                    msg=f"I0 for BSA by {app_name} between 60 and 65",
                )
                self.assertTrue(
                    float(result[index["I(0) StDev"]]) < 5,
                    msg=f"I0 std. for BSA by {app_name} below 5",
                )
                self.assertTrue(
                    float(result[index["First point"]]) < 70,
                    msg=f"Guinier region start for BSA by {app_name} below 70",
                )
                self.assertTrue(
                    float(result[index["Last point"]]) > 85,
                    msg=f"Guinier region end for BSA by {app_name} above 85",
                )
                self.assertTrue(
                    float(result[index["Quality"]]) > 0.8,
                    msg=f"Fit quality BSA by {app_name} above 0.8",
                )
                self.assertTrue(
                    float(result[index["Aggregated"]]) < 0.05,
                    msg=f"Aggregation number for BSA by {app_name} below 0.05",
                )
            elif result[index["File"]] == str(self.sas_curve2_filename):
                self.assertTrue(
                    abs(float(result[index["Rg"]]) - 3.05) <= 0.2,
                    msg=f"Rg for BSA by {app_name} between 2.85 and 3.25",
                )
                self.assertTrue(
                    float(result[index["Rg StDev"]]) < 0.15,
                    msg=f"Rg std. for SASDF52 by {app_name} below 0.5",
                )
                self.assertTrue(
                    abs(float(result[index["I(0)"]]) - 64.5) <= 2.5,
                    msg=f"I0 for SASDF52 by {app_name} between 62 and 67",
                )
                self.assertTrue(
                    float(result[index["I(0) StDev"]]) < 5,
                    msg=f"I0 std. for SASDF52 by {app_name} below 5",
                )
                self.assertTrue(
                    float(result[index["First point"]]) < 35,
                    msg=f"Guinier region start for SASDF52 by {app_name} below 35",
                )
                self.assertTrue(
                    float(result[index["Last point"]]) > 80,
                    msg=f"Guinier region end for SASDF52 by {app_name} above 80",
                )
                self.assertTrue(
                    float(result[index["Quality"]]) > 0.8,
                    msg=f"Fit quality SASDF52 by {app_name} above 0.8",
                )
                self.assertTrue(
                    float(result[index["Aggregated"]]) < 0.1,
                    msg=f"Aggregation number for SASDF52 by {app_name} below 0.1",
                )

    def test_ssv_output_with_two_files(self):
        """Test wether the output in the ssv table has the correct dimensions"""
        app_name: str = self.extra_arg["app"]
        test_output_name = pathlib.Path(self.cwd, f"{app_name}.ssv")
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
                str(test_output_name),
                "-f",
                "ssv",
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
            b"",
            msg=f"{app_name} for 2 files to ssv provided no output to stdout",
        )
        with open(str(test_output_name), "r") as test_output_file:
            test_output_result = test_output_file.readlines()
        test_output_result = [line[:-1].split() for line in test_output_result]

        self.assertEqual(
            len(test_output_result),
            2,
            msg=f"results created by {app_name} in native have the expected number of rows",
        )
        self.assertTrue(
            len(test_output_result[0]) == 9
            and len(test_output_result[1]) == 9,
            msg=f"results created by {app_name} in ssv have the expected number of elements",
        )
        self.assertEqual(
            {str(self.bsa_filename), str(self.sas_curve2_filename)},
            {
                test_output_result[0][-1],
                test_output_result[1][-1],
            },
            msg=f"ssv file created by {app_name} contains the filenames at the end of each row",
        )

    def test_native_output_with_two_files(self):
        """Test wether the output in the antive format has the correct dimensions"""
        app_name: str = self.extra_arg["app"]
        test_output_name = pathlib.Path(self.cwd, f"{app_name}.txt")
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
                str(test_output_name),
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode,
            0,
            msg=f"{app_name} for 2 files to native completed well",
        )
        self.assertEqual(
            run_app.stdout,
            b"",
            msg=f"{app_name} for 2 files to native provided no output to stdout",
        )
        with open(str(test_output_name), "r") as test_output_file:
            test_output_result = test_output_file.readlines()
        test_output_result = [line[:-1].split() for line in test_output_result]

        self.assertEqual(
            len(test_output_result),
            2,
            msg=f"results created by {app_name} in native have the expected number of rows",
        )
        self.assertTrue(
            len(test_output_result[0]) == 5
            and len(test_output_result[1]) == 5,
            msg=f"results created by {app_name} in native have the expected number of elements",
        )
        self.assertEqual(
            {str(self.bsa_filename), str(self.sas_curve2_filename)},
            {
                test_output_result[0][0],
                test_output_result[1][0],
            },
            msg=f"native file created by {app_name} contains "
            "the filenames at the start of each row",
        )

    def test_rg_of_SASDFX7_is_23Å(self):
        """
        SASDFX7 is an example of a dat file with artefacts at small angles.
        """
        app_name: str = self.extra_arg["app"]
        test_output_name = pathlib.Path(self.cwd, f"SASDFX7_{app_name}.csv")
        try:
            test_output_name.unlink()
        except FileNotFoundError:
            pass

        run_app = run(
            [
                app_name,
                str(self.SASDFX7),
                "-o",
                str(test_output_name),
                "-f",
                "csv",
                "-u",
                "Å",
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode,
            0,
            msg=f"{app_name} for SASDFX7 completed well",
        )

        with open(str(test_output_name), "r") as test_output_file:
            test_output_header = test_output_file.readline()[:-1].split(",")
            test_output_result = test_output_file.readline()
        result = test_output_result[:-1].split(",")

        expected_header = {
            "File",
            "Rg",
            "Rg StDev",
            "I(0)",
            "I(0) StDev",
            "First point",
            "Last point",
            "Quality",
            "Aggregated",
        }

        index = {}
        for header_item in expected_header:
            index[header_item] = test_output_header.index(header_item)

        self.assertTrue(
            abs(float(result[index["Rg"]]) - 2.3) <= 0.1,
            msg=f"Rg for SASDFX7 by {app_name} between 2.2 and 2.4",
        )


def suite():
    test_suite = unittest.TestSuite()
    for app in ["free_gpa", "free_guinier", "free_rg"]:
        if system() == "Windows":
            app += ".exe"
        test_suite.addTest(
            TestFreeSAS("test_one_bm29_bsa_without_arguments", app=app)
        )
        test_suite.addTest(
            TestFreeSAS("test_two_easy_files_without_arguments", app=app)
        )
        test_suite.addTest(
            TestFreeSAS("test_csv_output_with_two_files", app=app)
        )
        test_suite.addTest(
            TestFreeSAS("test_ssv_output_with_two_files", app=app)
        )
        test_suite.addTest(
            TestFreeSAS("test_native_output_with_two_files", app=app)
        )
        test_suite.addTest(TestFreeSAS("test_rg_of_SASDFX7_is_23Å", app=app))
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
