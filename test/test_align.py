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
    model.atoms = mol

class TestAlign(unittest.TestCase):
    testfile1 = join(base, "testdata", "dammif-01.pdb")
    testfile2 = join(base, "testdata", "dammif-02.pdb")
    
    def test_alignment(self):
        m = assign_model(self.testfile1)
        n = assign_model(self.testfile1)
        move(n)
        n.centroid()
        n.inertiatensor()
        n.canonical_position()
        n.centroid()
        n.inertiatensor()
        dist = alignment(m,n)
        self.assertAlmostEqual(dist, 0, 12, "bad alignment")

    def test_chg_position(self):
        m = assign_model(self.testfile1)
        n = assign_model(self.testfile2)
        dist_align = alignment(m,n)
        dist_ext = m.dist(n)
        self.assertEqual(dist_align, dist_ext, "molecule 2 unaligned")

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
        print n.enantiomer
        mol1_can = m.transform(m.can_param)
        mol2_can = n.transform(n.can_param)
        dist_before = m.dist(n, mol1_can, mol2_can)
        print dist_before
        symmetry = alignment(m,n)
        print symmetry
        mol2_sym = n.transform(n.can_param, symmetry)
        dist_after = m.dist(n, mol1_can, mol2_sym)
        print dist_after
        self.assertGreaterEqual(dist_before, dist_after, "increase of distance after alignment %s<%s with %s"%(dist_before, dist_after, symmetry))

def test_suite_all_alignment():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAlign("test_usefull_alignment"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_alignment()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)