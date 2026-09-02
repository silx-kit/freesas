"""End to end tests for auto_gpa.py """

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "02/09/2026"

import unittest
import pathlib
import logging
import shutil
import tempfile
from platform import system
from subprocess import run, PIPE, STDOUT
from os import linesep
from os.path import normpath
import codecs
import parse
from numpy import loadtxt
from freesas.test.utilstest import get_datafile

logger = logging.getLogger(__name__)

if system() == "Windows":
    free_bift = "free_bift.exe"
else:
    free_bift = "free_bift"


class TestBIFT(unittest.TestCase):
    """End to end tests for free_bift"""

    test_location = pathlib.Path(__file__)
    test_data_location = pathlib.Path(test_location.parent, "e2etest_data")
    bsa_filename = pathlib.Path(get_datafile("bsa_005_sub.dat"))
    sas_curve2_filename = pathlib.Path(get_datafile("SASDF52.dat"))
    SASDFX7 = pathlib.Path(get_datafile("SASDFX7.dat"))

    @classmethod
    def setUpClass(cls):
        """Send the output to a directory of our own, so that neither the
        current directory nor the directory holding the downloaded test data
        gets polluted."""
        super().setUpClass()
        cls.output_dir = pathlib.Path(tempfile.mkdtemp(prefix="e2etest_bift_"))
        cls.output_template = str(cls.output_dir / "{basename}.out")
        cls.expected_outfile_name_bsa = pathlib.Path(
            cls.output_dir, cls.bsa_filename.name
        ).with_suffix(".out")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir, ignore_errors=True)
        super().tearDownClass()

    def __init__(self, testName, **extra_kwargs):
        super().__init__(testName)
        self.extra_arg = extra_kwargs

    def remove_output_files(self):
        try:
            self.expected_outfile_name_bsa.unlink()
        except FileNotFoundError:
            pass

    def setUp(self):
        self.remove_output_files()
        return super().setUp()

    def tearDown(self):
        self.remove_output_files()
        return super().tearDown()

    def test_bm29_bsa_without_arguments_creates_out_file(self):
        """
        Test whether bift app on BSA data from BM29 creates an out file.
        """

        run_app = run(
            [
                free_bift,
                "--output",
                self.output_template,
                normpath(str(self.bsa_filename)),
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )
        self.assertEqual(
            run_app.returncode, 0, msg="bift on BM29 BSA completed well"
        )
        self.assertTrue(
            self.expected_outfile_name_bsa.exists(),
            f"bift on BM29 BSA created out file with correct name: {str(self.expected_outfile_name_bsa)}",
        )

    def test_bm29_bsa_out_file_has_the_expected_format(self):
        """
        Test whether bift app on BSA data from BM29 creates an out file.
        """

        _ = run(
            [
                free_bift,
                "--output",
                self.output_template,
                normpath(str(self.bsa_filename)),
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )

        with open(
            self.expected_outfile_name_bsa, "r", encoding="utf-8"
        ) as out_file:
            out_file_content = out_file.readlines()
        with codecs.open(
            self.expected_outfile_name_bsa, encoding="utf-8"
        ) as filecp:
            data = loadtxt(
                filecp,
                dtype=float,
                delimiter="\t",
                skiprows=9,
            )

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
            data.shape[1],
            3,
        )

    def test_bm29_bsa_result_numerically_matches_expectations(self):
        """
        Test whether the results of the bift app on BM29 BSA give roughly the
        expected Dmax, I₀ anr Rg and that the first is and the last point is close to 0.
        """

        run_app_ = run(
            [
                free_bift,
                "--output",
                self.output_template,
                normpath(str(self.bsa_filename)),
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )

        with open(
            self.expected_outfile_name_bsa, "r", encoding="utf-8"
        ) as out_file:
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

    def test_free_bift_outputs_one_line_summary(self):
        """
        Test whether free_bift app on BM29 BSA puts a one line summary in stdout.
        """

        run_app = run(
            [
                free_bift,
                "--output",
                self.output_template,
                normpath(str(self.bsa_filename)),
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )

        if system() == "Windows":
            run_app_output = str(run_app.stdout, encoding="utf-16")[:-1].replace("\\\\", "\\")
        else:
            run_app_output = str(run_app.stdout, encoding="utf-8")[:-1]
        run_app_output_parsed = parse.parse(
            str(self.expected_outfile_name_bsa)
            + ": Dmax= {Dmax}±{Dmax_err}; 𝛂= {alpha}±{alpha_err}; S₀= {S0}±{S0_err}; χ²= {chi_squared}±{chi_squared_err}; logP= {logP}±{logP_err}; Rg= {Rg}±{Rg_err}; I₀= {I0}±{I0_err}",
            run_app_output,
        )
        self.assertListEqual(
            list(run_app_output_parsed.named),
            [
                "Dmax",
                "Dmax_err",
                "alpha",
                "alpha_err",
                "S0",
                "S0_err",
                "chi_squared",
                "chi_squared_err",
                "logP",
                "logP_err",
                "Rg",
                "Rg_err",
                "I0",
                "I0_err",
            ],
            msg="Could not parse free_bift std output",
        )

    def test_free_bift_values_of_one_line_summary_match_expectations(self):
        """
        Test whether the one line summary of free_bift app on BM29 BSA gives the expected values.
        """

        run_app = run(
            [
                free_bift,
                "--output",
                self.output_template,
                normpath(str(self.bsa_filename)),
            ],
            stdout=PIPE,
            stderr=STDOUT,
            check=True,
        )

        if system() == "Windows":
            run_app_output = str(run_app.stdout, encoding="utf-16")[:-1].replace("\\\\", "\\")
        else:
            run_app_output = str(run_app.stdout, encoding="utf-8")[:-1]
        run_app_output_parsed = parse.parse(
            str(self.expected_outfile_name_bsa)
            + ": Dmax= {Dmax}±{Dmax_err}; 𝛂= {alpha}±{alpha_err}; S₀= {S0}±{S0_err}; χ²= {chi_squared}±{chi_squared_err}; logP= {logP}±{logP_err}; Rg= {Rg}±{Rg_err}; I₀= {I0}±{I0_err}",
            run_app_output,
        )
        self.assertAlmostEqual(
            float(run_app_output_parsed["Dmax"]),
            9.75,
            places=1,
            msg=f"expected Dmax to be close to 0.75 got {run_app_output_parsed['Dmax']}",
        )

        self.assertAlmostEqual(
            float(run_app_output_parsed["Rg"]),
            3.0,
            places=1,
            msg=f"expected Rg to be close to 3.0 got {run_app_output_parsed['Rg']}",
        )

        self.assertAlmostEqual(
            0.1 * float(run_app_output_parsed["I0"]),
            6.1,
            places=1,
            msg=f"expected I0 to be close to 60 got {run_app_output_parsed['I0']}",
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
    test_suite.addTest(TestBIFT("test_free_bift_outputs_one_line_summary"))
    test_suite.addTest(
        TestBIFT(
            "test_free_bift_values_of_one_line_summary_match_expectations"
        )
    )
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
