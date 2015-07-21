#!/usr/bin/python
__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import numpy
import unittest
from utilstests import base, join
from freesas.model import SASModel
from freesas.average import Grid

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignModels_test")


class TestAverage(unittest.TestCase):
    testfile1 = join(base, "testdata", "model-01.pdb")
    testfile2 = join(base, "testdata", "model-02.pdb")
    inputfiles = [testfile1, testfile2]
    grid = Grid(inputfiles)

    def test_gridsize(self):
        inputfiles = self.inputfiles
        grid = self.grid
        size = grid.spatial_extent()
        coordmax = numpy.array([size[0:3]], dtype="float")
        coordmin = numpy.array([size[3:6]], dtype="float")

        pb = 0
        for i in inputfiles:
            m = SASModel(i)
            a = coordmin + m.atoms[:, 0:3]
            b = m.atoms[:, 0:3] - coordmax
            if (a >= 0.0).any() or (b >= 0.0).any():
                pb += 1
        self.assertEqual(pb, 0, msg="computed size is not the good one")

    def test_knots(self):
        grid = self.grid
        nbknots = numpy.random.randint(4000, 6000)
        threshold = 10.0#acceptable difference between nbknots and the effective nb of knots in percent
        grid.calc_radius(nbknots)
        grid.make_grid()
        gap = (1.0 * (grid.nbknots - nbknots) / nbknots) * 100
        self.assertGreater(threshold, gap, msg="final number of knots too different of wanted number: %s != %s"%(nbknots, grid.nbknots))

    def test_makegrid(self):
        grid = self.grid
        lattice = grid.make_grid()
        m = SASModel(lattice)
        self.assertAlmostEqual(m.fineness, 2*grid.radius, 10, msg="grid do not have the computed radius")

def test_suite_all_average():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAverage("test_gridsize"))
    testSuite.addTest(TestAverage("test_knots"))
    testSuite.addTest(TestAverage("test_makegrid"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_average()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
