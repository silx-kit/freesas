#!/usr/bin/python
__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import numpy
import unittest
import os
import tempfile
from utilstests import base, join
from freesas.model import SASModel

class TesttParser(unittest.TestCase):
    testfile = join(base, "testdata", "model-01.pdb")

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.tmpdir = tempfile.mkdtemp()
        self.outfile = join(self.tmpdir, "out.pdb")

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        for fn in (self.outfile,self.tmpdir):
            if os.path.exists(fn):
                if os.path.isdir(fn):
                    os.rmdir(fn)
                else:
                    os.unlink(fn)

    def test_same(self):
        m = SASModel()
        m.read(self.testfile)
        m.save(self.outfile)
        infile = open(self.testfile).read()
        outfile = open(self.outfile).read()
        self.assertEqual(infile, outfile, msg="file content is the same")

    def test_dist(self):
        molecule = numpy.random.randint(0,100, size=400).reshape(100,4).astype(float)
        molecule[:,-1] = 1.0
        m = SASModel(molecule*1.0)
        n = SASModel(molecule*1.0)
        distance = m.dist(n, m.atoms, n.atoms)
        self.assertEqual(distance, 0, msg="NSD different of 0: %s!=0"%(distance))

    def test_can_transform(self):
        molecule = numpy.random.randint(0,100, size=400).reshape(100,4).astype(float)
        molecule[:,-1] = 1.0
        m = SASModel(molecule*1.0)
        m.centroid()
        m.inertiatensor()
        m.canonical_parameters()
        p0 = m.can_param
        mol1 = m.transform(p0,[1,1,1])
        assert abs(mol1-molecule).max() != 0 ,"molecule did not move"
        m.atoms = mol1
        m.centroid()
        m.inertiatensor()
        com = m.com
        tensor = m.inertensor
        diag = numpy.eye(3)
        matrix = tensor-tensor*diag
        self.assertAlmostEqual(abs(com).sum(), 0, 10, msg="molecule not on its center of mass")
        self.assertAlmostEqual(abs(matrix).sum(), 0, 10, "inertia moments unaligned ")

    def test_dist_move(self):
        molecule = numpy.random.randint(0,100, size=400).reshape(100,4).astype(float)
        molecule[:,-1] = 1.0
        m = SASModel(molecule*1.0)
        n = SASModel(molecule*1.0)
        m.centroid()
        m.inertiatensor()
        m.canonical_parameters()
        n.centroid()
        n.inertiatensor()
        n.canonical_parameters()
        assert abs(n.atoms-m.atoms).max()==0, "molecules are different"
        p0 = m.can_param
        dist_after_mvt = m.dist_after_movement(n, p0, [1,1,1])
        self.assertEqual(dist_after_mvt, 0, msg="NSD different of 0: %s!=0"%(dist_after_mvt))
        
def test_suite_all_model():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TesttParser("test_same"))
    testSuite.addTest(TesttParser("test_dist"))
    testSuite.addTest(TesttParser("test_can_transform"))
    testSuite.addTest(TesttParser("test_dist_move"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_model()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)