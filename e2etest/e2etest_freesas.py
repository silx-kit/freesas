# -*- coding: utf-8 -*-
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2017  European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__authors__ = ["Martha Brennich"]
__license__ = "MIT"
__date__ = "11/07/2020"

import unittest
import pathlib
import re
import logging
from subprocess import run, Popen, PIPE, STDOUT
from os import linesep

logger = logging.getLogger(__name__)

expectedTexts = {
    "Label of scatter plot X-axis": r"\$q\$ \(nm\$\^\{-1\}\$\)",
    "Label of scatter plot Y-axis": r"\$I\(q\)\$ \(log scale\)",
    "Smallest Tick of scatter plot Y axis": r"\$\\mathdefault\{10\^\{-2\}\}\$",
    "Largest Tick of scatter plot Y axis": r"\$\\mathdefault\{10\^\{2\}\}\$",
    "Scattering plot caption": r"Scattering curve",
    "Experimental data legend": r"Experimental data", #used twice, but we might just ignore that
    "Guinier region legend": r"Guinier region: \$R_g=\$[23]\.[0-9][0-9] nm, \$I_0=\$6[0-9]\.[0-9][0-9]",
    "BIFT fit legend": r"BIFT extraplolated: \$D_\{max\}=\$9\.[0-9][0-9] nm",
    "Label of Guinier plot X-axis": r"\$q\^2\$ \(nm\$\^\{-2\}\$\)",
    "Label of Guinier plot Y-axis": r"ln\[\$I\(q\)\$]",
    "Guinier - qRgmin": r"\$\(qR_\{g\}\)_\{min\}\$=0\.[0-9]",
    "Guinier - qRgmax": r"\$\(qR_\{g\}\)_\{max\}\$=1\.[0123]",
    "Guinier region label": r"Guinier region",
    "Guinier plot caption": r"Guinier plot: \$R_\{g\}=\$[23]\.[0-9][0-9] nm \$I_\{0\}=\$6[0-9]\.[0-9][0-9]",
    "Guinier fit equation": r"ln\[\$I\(q\)\$\] = 4\.12 -3\.01 \* \$q\^2\$",
    "Guinier fit data label": r"Experimental curve",
    "Label of Kratky plot X-Axis":r"\$qR_\{g\}\$",
    "Label of Kratky plot Y-Axis":r"\$\(qR_\{g\}\)\^2 I/I_\{0\}\$",
    "Kratky plot caption": r"Dimensionless Kratky plot",
    "Label of distribution plot X-axis": r"\$r\$ \(nm\)",
    "Label of distribution plot Y-axis": r"\$p\(r\)\$",
    "Distribution plot caption": r"Pair distribution function",
    "BIFT chi": r"BIFT: χ\$_\{r\}\^\{2\}=\$1\.[0-9][0-9]",
    "BIFT Dmax": r"\$D_\{max\}=\$[1]?[09].[0-9][0-9] nm",
    "BIFT Rg": r"\$R_\{g\}=\$[23]\.[0-9][0-9] nm",
    "BIFT I0": r"\$I_\{0\}=\$6[0-9]\.[0-9][0-9]",
}


class TestFreeSAS(unittest.TestCase):

    cwd = pathlib.Path.cwd()
    TEST_IMAGE_NAME = pathlib.Path(cwd, "freesas.svg")
    test_location = pathlib.Path(__file__)
    test_data_location = pathlib.Path(test_location.parent, "e2etest_data")
    bsa_filename = pathlib.Path(test_data_location, "bsa_005_sub.dat")
    image_text = None

    @classmethod
    def setUpClass(cls):
        super(TestFreeSAS, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestFreeSAS, cls).tearDownClass()
        cls.image_text = None

    def __init__(self, testName, **extra_kwargs):
        super(TestFreeSAS, self).__init__(testName)
        self.extra_arg = extra_kwargs

    def test_save_image(self):
        """
        Test whether freeSAS finishes without errors
        if there is an -o argument.
        It also uses the output as input for label tests.
        """
        #Make sure the result file does not exist for a meaningful assert
        try:
            self.TEST_IMAGE_NAME.unlink()
        except FileNotFoundError:
            pass
        run_freesas = run(["freesas", str(self.bsa_filename),
                           "-o", str(self.TEST_IMAGE_NAME)],
                          stdout=PIPE, stderr=STDOUT, check=True)
        self.assertEqual(run_freesas.returncode, 0, msg="freesas completed well")
        self.assertTrue(self.TEST_IMAGE_NAME.exists(), msg="Found output file")
        with open(self.TEST_IMAGE_NAME) as file:
            self.__class__.image_text = file.read()
        try:
            self.TEST_IMAGE_NAME.unlink()
        except FileNotFoundError:
            pass

    def test_display_image(self):
        """
        Test whether freeSAS for one dataset finishes without errors
        if there no -o argument.
        """
        run_freesas = Popen(["freesas", str(self.bsa_filename)],
                            universal_newlines=True,
                            stdout=PIPE, stderr=PIPE, stdin=PIPE)
        stdout, _ = run_freesas.communicate(linesep, timeout=20)
        self.assertEqual(run_freesas.returncode, 0,
                         msg="freesas completed well")
        self.assertEqual(stdout, "Press enter to quit",
                         msg="freesas requested enter")

    def test_label(self):
        """
        Test for the presence of labels in the svg.
        Requires two extra kwargs:
        regex: The regex expression to search for
        description: The description of what the label represents
        """
        text_regex: str = self.extra_arg["regex"]
        text_description: str = self.extra_arg["description"]
        pattern = re.compile(text_regex)
        self.assertIsNotNone(
            pattern.search(self.image_text),
            msg="Could not find text for {} in image".format(text_description)
        )

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestFreeSAS("test_display_image"))
    test_suite.addTest(TestFreeSAS("test_save_image"))
    for text_description, text_regex in expectedTexts.items():
        test_suite.addTest(TestFreeSAS("test_label",
                                       regex=text_regex,
                                       description=text_description))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
