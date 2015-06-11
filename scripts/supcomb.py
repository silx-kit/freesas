#!/usr/bin/python
from __builtin__ import False
__author__ = "Guillaume Bonamis"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import argparse
from os.path import dirname, abspath
base = dirname(dirname(abspath(__file__)))
from freesas.align import alignment_2models

parser = argparse.ArgumentParser(description="align several models and calculate NSD")
parser.add_argument("file1", help="pdb file")
parser.add_argument("file2", help="second pdb file")
parser.add_argument("-f", "--fast", action="store_false", help="using fast mode, default=slow")
parser.add_argument("-e", "--enantiomorphs", action="store_false", help="do not search enantiomorphs, default=search")

args = parser.parse_args()
dist = alignment_2models(args.file1, args.file2, enantiomorphs=args.enantiomorphs, slow=args.fast)
print "NSD after optimized alignment = %s"%(dist)