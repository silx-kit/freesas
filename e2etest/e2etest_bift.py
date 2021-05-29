"""End to end tests for auto_gpa.py """

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "27/12/2020"

import unittest
import pathlib
import logging
from system import platform
from subprocess import run, PIPE, STDOUT
from os import linesep
from os.path import normpath
from numpy import loadtxt

logger = logging.getLogger(__name__)


class TestBIFT(unittest.TestCase):
    """End to end tests for free_bift"""

    cwd = pathlib.Path.cwd()
    test_location = pathlib.Path(__file__)
    test_data_location = pathlib.Path(test_location.parent, "e2etest_data")
    bsa_filename = pathlib.Path(test_data_location, "bsa_005_sub.dat")
    sas_curve2_filename = pathlib.Path(test_data_location, "SASDF52.dat")
    SASDFX7 = pathlib.Path(test_data_location, "SASDFX7.dat")

    def __init__(self, testName, **extra_kwargs):
        super().__init__(testName)
        self.extra_arg = extra_kwargs

    def test_bm29_bsa_without_arguments_creates_out_file(self):
        """
        Test whether bift app on BSA data from BM29 creates an out file.
        """
        expected_outfile_name = pathlib.Path(
            self.cwd, self.bsa_filename.name
        ).with_suffix(".out")
        try:
            expected_outfile_name.unlink()
        except FileNotFoundError:
            pass
        if s
        run_app = run(
            [free_bift, normpath(str(self.bsa_filename))],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode, 0, msg="bift on BM29 BSA completed well"
        )
        self.assertTrue(
            expected_outfile_name.exists(),
            f"bift on BM29 BSA created out file with correct name: {str(expected_outfile_name)}",
        )

    def test_bm29_bsa_out_file_has_the_expected_format(self):
        """
        Test whether bift app on BSA data from BM29 creates an out file.
        """
        expected_outfile_name = pathlib.Path(
            self.cwd, self.bsa_filename.name
        ).with_suffix(".out")
        try:
            expected_outfile_name.unlink()
        except FileNotFoundError:
            pass
        run_app = run(
            ["free_bift", normpath(str(self.bsa_filename))],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode, 0, msg="bift on BM29 BSA completed well"
        )
        with open(expected_outfile_name, "r") as out_file:
            out_file_content = out_file.readlines()

        self.assertEqual(out_file_content[0].strip(), f"# {self.bsa_filename}")
        self.assertTrue(
            out_file_content[1].startswith("# Dmax= ")
            and "±" in out_file_content[1],
            msg=f"exptexted line to resemble # Dmax= 9.76±0.05 got {out_file_content[1]}",
        )
        self.assertTrue(
            out_file_content[2].startswith("# 𝛂= ")
            and "±" in out_file_content[2],
            msg=f"exptexted line to resemble # 𝛂= 8764.5±1384.2 got {out_file_content[2]}",
        )
        self.assertTrue(
            out_file_content[3].startswith("# S₀= ")
            and "±" in out_file_content[3],
            msg=f"exptexted line to resemble # S₀= 0.0002±0.0000  got {out_file_content[3]}",
        )
        self.assertTrue(
            out_file_content[4].startswith("# χ²= ")
            and "±" in out_file_content[4],
            msg=f"exptexted line to resemble # χ²= 1.89±0.00 got {out_file_content[4]}",
        )
        self.assertTrue(
            out_file_content[5].startswith("# logP= ")
            and "±" in out_file_content[5],
            msg=f"exptexted line to resemble # logP= -914.15±0.47 got {out_file_content[5]}",
        )
        self.assertTrue(
            out_file_content[6].startswith("# Rg= ")
            and "±" in out_file_content[6],
            msg=f"exptexted line to resemble # Rg= 2.98±0.00  got {out_file_content[6]}",
        )
        self.assertTrue(
            out_file_content[7].startswith("# I₀= ")
            and "±" in out_file_content[7],
            msg=f"exptexted line to resemble 60.86±0.00 got {out_file_content[7]}",
        )
        self.assertEqual(out_file_content[8].strip(), "")
        self.assertEqual(
            out_file_content[9].strip(),
            "# r\tp(r)\tsigma_p(r)",
        )
        self.assertEqual(
            loadtxt(
                expected_outfile_name, dtype=float, delimiter="\t", skiprows=9
            ).shape[1],
            3,
        )

    def test_bm29_bsa_result_numerically_matches_expectations(self):
        """
        Test whether the results of the bift app on BM29 BSA give roughly the
        expected Dmax, I₀ anr Rg and that the first is and the last point is close to 0.
        """
        expected_outfile_name = pathlib.Path(
            self.cwd, self.bsa_filename.name
        ).with_suffix(".out")
        try:
            expected_outfile_name.unlink()
        except FileNotFoundError:
            pass
        run_app = run(
            ["free_bift", normpath(str(self.bsa_filename))],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode, 0, msg="bift on BM29 BSA completed well"
        )
        with open(expected_outfile_name, "r") as out_file:
            out_file_content = out_file.readlines()

        self.assertAlmostEqual(
            float(out_file_content[1][8:12]),
            9.75,
            places=1,
            msg=f"expected Dmax to be close to 0.75 got {out_file_content[1]}",
        )

        self.assertAlmostEqual(
            float(out_file_content[6][6:10]),
            3.0,
            places=1,
            msg=f"expected Rg to be close to 3.0 got {out_file_content[6]}",
        )

        self.assertAlmostEqual(
            0.1 * float(out_file_content[7][6:10]),
            6.1,
            places=1,
            msg=f"expected I0 to be close to 60 got {out_file_content[7]}",
        )

        self.assertEqual(
            out_file_content[10].strip(),
            "0.0\t0.0\t0.0",
            msg=f"Expected first p(r) line to be '0.0     0.0     0.0' got {out_file_content[10]}",
        )

        last_line_content = out_file_content[-1].split("\t")

        self.assertAlmostEqual(
            float(last_line_content[0]),
            9.75,
            places=1,
            msg=f"expected last r point to be close to 9.75 got {last_line_content[0]}",
        )

        self.assertAlmostEqual(
            float(last_line_content[1]),
            0,
            places=2,
            msg=f"expected last r point to be close to 0 got {last_line_content[1]}",
        )


def suite():
    """Build test suite for free_bift"""
    test_suite = unittest.TestSuite()

    test_suite.addTest(
        TestBIFT("test_bm29_bsa_without_arguments_creates_out_file")
    )
    test_suite.addTest(
        TestBIFT("test_bm29_bsa_out_file_has_the_expected_format")
    )
    test_suite.addTest(
        TestBIFT("test_bm29_bsa_result_numerically_matches_expectations")
    )
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
