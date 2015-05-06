#!/usr/bin/python
__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF" 

import numpy
import unittest

import sys, os
from os.path import dirname, abspath, join
base = dirname(dirname(abspath(__file__)))
if base not in sys.modules:
    sys.path.insert(0, base)
import tempfile
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

def test_suite_all_model():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TesttParser("test_same"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_model()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)