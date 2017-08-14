#!/usr/bin/python
# coding: utf-8
from __future__ import print_function

__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import numpy
import unittest
from .utilstests import get_datafile
from ..model import SASModel
from ..average import Grid, AverModels

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignModels_test")


class TestAverage(unittest.TestCase):
    testfile1 = get_datafile("model-01.pdb")
    testfile2 = get_datafile("model-02.pdb")
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
        threshold = 10.0  # acceptable difference between nbknots and the effective nb of knots in percent
        grid.calc_radius(nbknots)
        grid.make_grid()
        gap = (1.0 * (grid.nbknots - nbknots) / nbknots) * 100
        self.assertGreater(threshold, gap, msg="final number of knots too different of wanted number: %s != %s" % (nbknots, grid.nbknots))

    def test_makegrid(self):
        grid = self.grid
        lattice = grid.make_grid()
        m = SASModel(lattice)
        self.assertAlmostEqual(m.fineness, 2 * grid.radius, 10, msg="grid do not have the computed radius")

    def test_read(self):
        inputfiles = self.inputfiles
        average = AverModels(inputfiles, self.grid.coordknots)
        models = [SASModel(inputfiles[1]), SASModel(inputfiles[0])]
        average.read_files(reference=1)
        diff = 0.0
        for i in range(len(inputfiles)):
            diff += (models[i].atoms - average.models[i].atoms).max()
        self.assertAlmostEqual(diff, 0.0, 10, msg="Files not read properly")

    def test_occupancy(self):
        average = AverModels(self.inputfiles, self.grid.coordknots)
        average.read_files()
        occ_grid = average.assign_occupancy()
        average.grid = occ_grid
        assert occ_grid.shape[-1] == 5, "problem in grid shape"
        diff = occ_grid[:-1, 3] - occ_grid[1:, 3]
        self.assertTrue(diff.max() >= 0.0, msg="grid is not properly sorted with occupancy")


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAverage("test_gridsize"))
    testSuite.addTest(TestAverage("test_knots"))
    testSuite.addTest(TestAverage("test_makegrid"))
    testSuite.addTest(TestAverage("test_read"))
    testSuite.addTest(TestAverage("test_occupancy"))
    return testSuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
