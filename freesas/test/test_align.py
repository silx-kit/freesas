#!/usr/bin/python
from __future__ import print_function

__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import numpy
import unittest
from .utilstests import get_datafile
from ..model import SASModel
from ..align import AlignModels
from ..transformations import translation_matrix, euler_matrix
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignModels_test")


def move(mol):
    """
    Random movement of the molecule.

    @param mol: 2d array, coordinates of the molecule
    @return mol:2D array, coordinates of the molecule after a translation and a rotation
    """
    vect = numpy.random.random(3)
    translation = translation_matrix(vect)

    euler = numpy.random.random(3)
    rotation = euler_matrix(euler[0], euler[1], euler[2])

    mol = numpy.dot(rotation, mol.T)
    mol = numpy.dot(translation, mol).T

    return mol


def assign_random_mol(inf=None, sup=None):
    """
    Create a random 2d array to create a molecule

    @param inf: inf limit of coordinates values
    @param sup: sup limit of coordinates values
    @return molecule: 2d array, random coordinates
    """
    if not inf:
        inf = 0
    if not sup:
        sup = 100
    molecule = numpy.random.randint(inf, sup, size=400).reshape(100, 4).astype(float)
    molecule[:, -1] = 1.0
    return molecule


class TestAlign(unittest.TestCase):
    testfile1 = get_datafile("dammif-01.pdb")
    testfile2 = get_datafile("dammif-02.pdb")

    def test_alignment(self):
        inputfiles = [self.testfile1, self.testfile1]
        align = AlignModels(inputfiles, slow=False)
        align.assign_models()
        m = align.models[0]
        n = align.models[1]
        n.atoms = move(n.atoms)
        n.centroid()
        n.inertiatensor()
        n.canonical_parameters()
        if m.dist(n, m.atoms, n.atoms) == 0:
            logger.error(m.dist(n, m.atoms, n.atoms))
            logger.error("pb of movement")
        dist = align.alignment_2models(save=False)
        self.assertAlmostEqual(dist, 0, 12, msg="NSD unequal 0, %s!=0" % dist)

    def test_usefull_alignment(self):
        inputfiles = [self.testfile1, self.testfile2]
        align = AlignModels(inputfiles, slow=False)
        align.assign_models()
        mol1 = align.models[0]
        mol2 = align.models[1]
        dist_before = mol1.dist(mol2, mol1.atoms, mol2.atoms)
        symmetry, par = align.alignment_sym(mol1, mol2)
        dist_after = mol1.dist_after_movement(par, mol2, symmetry)
        self.assertGreaterEqual(dist_before, dist_after, "increase of distance after alignment %s<%s" % (dist_before, dist_after))

    def test_optimisation_align(self):
        inputfiles = [self.testfile1, self.testfile2]
        align = AlignModels(inputfiles, slow=False)
        align.assign_models()
        mol1 = align.models[0]
        mol2 = align.models[1]
        align.slow = False
        sym0, p0 = align.alignment_sym(mol1, mol2)
        dist_before = mol1.dist_after_movement(p0, mol2, sym0)
        align.slow = True
        sym, p = align.alignment_sym(mol1, mol2)
        dist_after = mol1.dist_after_movement(p, mol2, sym)
        self.assertGreaterEqual(dist_before, dist_after, "increase of distance after optimized alignment %s<%s" % (dist_before, dist_after))

    def test_alignment_intruder(self):
        intruder = numpy.random.randint(0, 8)
        inputfiles = []
        for i in range(8):
            if i == intruder:
                inputfiles.append(self.testfile2)
            else:
                inputfiles.append(self.testfile1)

        align = AlignModels(inputfiles, slow=False, enantiomorphs=False)
        align.assign_models()
        align.validmodels = numpy.ones(8)
        table = align.makeNSDarray()
        if table.sum() == 0:
            logger.error("there is no intruders")

        averNSD = ((table.sum(axis=-1)) / (align.validmodels.sum() - 1))
        num_intr = averNSD.argmax()

        if not num_intr and num_intr != 0:
            logger.error("cannot find the intruder")
        self.assertEqual(num_intr, intruder, msg="not find the good intruder, %s!=%s" % (num_intr, intruder))

    def test_reference(self):
        inputfiles = [self.testfile1] * 8
        align = AlignModels(inputfiles, slow=False, enantiomorphs=False)
        align.assign_models()
        for i in range(8):
            mol = assign_random_mol()
            align.models[i].atoms = mol
        align.validmodels = numpy.ones(8)
        table = align.makeNSDarray()
        ref = align.find_reference()
        neg_dif = 0
        for i in range(8):
            dif = (table[i, :] - table[ref, :]).mean()
            if dif < 0:
                neg_dif += 1
        self.assertEqual(neg_dif, 0, msg="pb with reference choice")


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAlign("test_alignment"))
    testSuite.addTest(TestAlign("test_usefull_alignment"))
    testSuite.addTest(TestAlign("test_optimisation_align"))
    testSuite.addTest(TestAlign("test_alignment_intruder"))
    testSuite.addTest(TestAlign("test_reference"))
    return testSuite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
