#!/usr/bin/python
__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF" 

import numpy
import unittest
import sys, os
from utilstests import base, join
from freesas.model import SASModel
from freesas.align import alignment, assign_model
from random import uniform
from scipy.optimize import fmin

def move(model):
    """
    random movement of the molecule
    
    Parameters
    ----------
    model: SASModel
    
    Output
    ----------
    mol: 2D array, coordinates of the molecule after a translation and a rotation
    """
    mol = model.atoms
    mol = mol.T
    
    translation = numpy.identity(4, dtype="float")
    translation[0:3,3] = numpy.array([[uniform(-100,100),uniform(-100,100),uniform(-100,100)]], dtype="float")
    
    rotation = numpy.identity(4, dtype="float")
    x = uniform(-1,1)
    y = uniform(-1,1)
    z = uniform(-1,1)
    norme = (x*x+y*y+z*z)**0.5
    x = x/norme
    y = y/norme
    z = z/norme
    theta = uniform(0,2*numpy.pi)
    c = numpy.cos(theta)
    op = 1-c
    s = numpy.sin(theta)
    rotation[0,0]= x*x*op+c
    rotation[0,1]= x*y*op-z*s
    rotation[0,2]= x*z*op+y*s
    rotation[1,0]= y*x*op+z*s
    rotation[1,1]= y*y*op+c
    rotation[1,2]= y*z*op-x*s
    rotation[2,0]= z*x*op-y*s
    rotation[2,1]= z*y*op+x*s
    rotation[2,2]= z*z*op+c
    
    mol = numpy.dot(rotation, mol)
    mol = numpy.dot(translation, mol)
    
    mol = mol.T
    return mol

class TestAlign(unittest.TestCase):
    testfile1 = join(base, "testdata", "dammif-01.pdb")
    testfile2 = join(base, "testdata", "dammif-02.pdb")
    
    def test_alignment(self):
        molecule = numpy.random.randint(-100,0, size=400).reshape(100,4).astype(float)
        molecule[:,-1] = 1.0
        m = SASModel(molecule*1.0)
        n = SASModel(molecule*1.0)
        n.atoms = move(n)
        m.centroid()
        m.inertiatensor()
        m.canonical_parameters()
        param1 = m.can_param
        mol1_can = m.transform(param1,[1,1,1])
        n.centroid()
        n.inertiatensor()
        n.canonical_parameters()
        param2 = n.can_param
        mol2_can = n.transform(param2,[1,1,1])
        assert m.dist(n, mol1_can, mol2_can) != 0, "pb of movement"
        sym2 = alignment(m,n)
        mol2_align = n.transform(param2, sym2)
        dist = m.dist(n, mol1_can, mol2_align)
        self.assertAlmostEqual(dist, 0, 12, "bad alignment %s!=0"%(dist))

    def test_usefull_alignment(self):
        molecule1 = numpy.random.randint(-100,0, size=400).reshape(100,4).astype(float)
        molecule1[:,-1] = 1.0
        molecule2 = numpy.random.randint(-100,0, size=400).reshape(100,4).astype(float)
        molecule2[:,-1] = 1.0
        m = SASModel(molecule1*1.0)
        n = SASModel(molecule2*1.0)
        m.centroid()
        m.inertiatensor()
        m.canonical_parameters()
        n.centroid()
        n.inertiatensor()
        n.canonical_parameters()
        mol1_can = m.transform(m.can_param,[1,1,1])
        mol2_can = n.transform(n.can_param,[1,1,1])
        dist_before = m.dist(n, mol1_can, mol2_can)
        symmetry = alignment(m,n)
        mol2_sym = n.transform(n.can_param, symmetry)
        dist_after = m.dist(n, mol1_can, mol2_sym)
        self.assertGreaterEqual(dist_before, dist_after, "increase of distance after alignment %s<%s with %s"%(dist_before, dist_after, symmetry))

    def test_optimisation_align(self):
        molecule1 = numpy.random.randint(-100,0, size=400).reshape(100,4).astype(float)
        molecule1[:,-1] = 1.0
        molecule2 = numpy.random.randint(-100,0, size=400).reshape(100,4).astype(float)
        molecule2[:,-1] = 1.0
        m = SASModel(molecule1*1.0)
        n = SASModel(molecule2*1.0)
        m.centroid()
        m.inertiatensor()
        m.canonical_parameters()
        n.centroid()
        n.inertiatensor()
        n.canonical_parameters()
        p0 = n.can_param
        sym = alignment(m,n)
        dist_before = m.dist_after_movement(p0, n, sym)
        p = fmin(m.dist_after_movement, p0, args=(n, sym), maxiter=200)
        dist_after = m.dist_after_movement(p, n, sym)
        self.assertGreater(dist_before, dist_after, msg="distance is not optimised : %s<=%s"%(dist_before,dist_after))
        
def test_suite_all_alignment():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAlign("test_alignment"))
    testSuite.addTest(TestAlign("test_usefull_alignment"))
    testSuite.addTest(TestAlign("test_optimisation_align"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_alignment()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)