#!/usr/bin/python
__author__ = "Guillaume Bonamis"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import argparse
from os.path import dirname, abspath
base = dirname(dirname(abspath(__file__)))
from freesas.align import alignment_2models

usage = "./supcomb file1.pdb file2.pdb [OPTIONS]"
description = "align several models and calculate NSD"
parser = argparse.ArgumentParser(usage=usage, description=description)
parser.add_argument("file", metavar="FILE", nargs='+', help="pdb files to align")
parser.add_argument("-m", "--mode",dest="mode", type=str, choices=["SLOW", "FAST"], default="SLOW", help="Either SLOW or FAST, default: %(default)s)")
parser.add_argument("-e", "--enantiomorphs",type=str, choices=["YES", "NO"], default="YES", help="Search enantiomorphs, YES or NO, default: %(default)s)")

args = parser.parse_args()
if args.mode=="SLOW":
    slow = True
else:
    slow = False
if args.enantiomorphs=="YES":
    enantiomorphs = True
else:
    enantiomorphs = False
dist = alignment_2models(args.file[0], args.file[1], enantiomorphs=enantiomorphs, slow=slow)
print "%s and %s aligned"%(args.file[0], args.file[1])
print "NSD after optimized alignment = %s"%(dist)