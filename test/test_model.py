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
    
    def test_transform(self):
        molecule = numpy.random.randint(0,100, size=400).reshape(100,4).astype(float)
        molecule[:,-1] = 1.0
        m = SASModel(molecule*1.0)
        m.centroid()
        m.inertiatensor()
        m.canonical_parameters()
        p0 = m.can_param
        sym = m.enantiomer
        print p0
        print sym
        mol1 = m.transform(p0,sym)
        print abs(mol1-molecule).max()
        m.canonical_position()
        mol2 = m.atoms
        print abs(mol2-mol1).max()

def test_suite_all_model():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TesttParser("test_same"))
    testSuite.addTest(TesttParser("test_transform"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_model()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)