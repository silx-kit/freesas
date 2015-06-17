#!/usr/bin/python
__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF" 

import numpy
import unittest
from utilstests import base, join
from freesas.align import AlignModels
from freesas.transformations import translation_matrix, euler_matrix
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
    if not inf: inf = 0
    if not sup: sup = 100
    molecule = numpy.random.randint(inf ,sup, size=400).reshape(100,4).astype(float)
    molecule[:,-1] = 1.0
    return molecule

class TestAlign(unittest.TestCase):
    testfile1 = join(base, "testdata", "dammif-01.pdb")
    testfile2 = join(base, "testdata", "dammif-02.pdb")
    
    def test_alignment(self):
        m = assign_random_mol()
        n = move(m*1.0)
        align = AlignModels()
        align.assign_models(m)
        align.assign_models(n)
        mol1 = align.models[0]
        mol2 = align.models[1]
        if mol1.dist(mol2, m, n)==0:
            logger.error("pb of movement")
        dist = align.alignment_2models(save=False)
        self.assertAlmostEqual(dist, 0, 12, msg="NSD unequal 0, %s!=0"%dist)

    def test_usefull_alignment(self):
        m = assign_random_mol()
        n = assign_random_mol()
        align = AlignModels()
        align.assign_models(m)
        align.assign_models(n)
        mol1 = align.models[0]
        mol2 = align.models[1]
        dist_before = mol1.dist(mol2, mol1.atoms, mol2.atoms)
        symmetry, par = align.alignment_sym(mol1,mol2)
        dist_after = mol1.dist_after_movement(par, mol2, symmetry)
        self.assertGreaterEqual(dist_before, dist_after, "increase of distance after alignment %s<%s"%(dist_before, dist_after))

    def test_optimisation_align(self):
        m = assign_random_mol()
        n = assign_random_mol()
        align = AlignModels()
        align.assign_models(m)
        align.assign_models(n)
        mol1 = align.models[0]
        mol2 = align.models[1]
        align.slow = False
        sym0, p0 = align.alignment_sym(mol1,mol2)
        dist_before = mol1.dist_after_movement(p0, mol2, sym0)
        align.slow = True
        sym, p = align.alignment_sym(mol1,mol2)
        dist_after = mol1.dist_after_movement(p, mol2, sym)
        self.assertGreaterEqual(dist_before, dist_after, "increase of distance after optimized alignment %s<%s"%(dist_before, dist_after))

    def test_alignment_intruder(self):
        align = AlignModels()
        align.slow = False
        align.enantiomorphs = False
        m = assign_random_mol()
        intruder = numpy.random.randint(0, 8)
        
        for i in range(8):
            if i==intruder:
                mol = assign_random_mol()
                align.assign_models(mol)
            else:
                align.assign_models(m)
        table = align.makeNSDarray()
        if table.sum()==0:
            logger.error("there is no intruders")
        
        num_intr = None
        max_dist = 0.00
        for i in range(len(table)):
            aver = table[i,:].mean()
            if aver>=max_dist:
                max_dist = aver
                num_intr = i
        if not num_intr and num_intr!=0:
            logger.error("cannot find the intruder")
        self.assertEqual(num_intr, intruder, msg="not find the good intruder")

    def test_reference(self):
        align = AlignModels()
        align.slow = False
        align.enantiomorphs = False
        for i in range(8):
            mol = assign_random_mol()
            align.assign_models(mol)
        table = align.makeNSDarray()
        ref = align.find_reference()
        neg_dif = 0
        for i in range(8):
            dif = (table[i,:]-table[ref,:]).mean()
            if dif<0:
                neg_dif += 1
        self.assertEqual(neg_dif, 0, msg="pb with reference choice")
        
def test_suite_all_alignment():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAlign("test_alignment"))
    testSuite.addTest(TestAlign("test_usefull_alignment"))
    testSuite.addTest(TestAlign("test_optimisation_align"))
    testSuite.addTest(TestAlign("test_alignment_intruder"))
    testSuite.addTest(TestAlign("test_reference"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_alignment()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)