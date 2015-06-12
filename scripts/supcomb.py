#!/usr/bin/python
__author__ = "Guillaume Bonamis"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import argparse
from os.path import dirname, abspath
base = dirname(dirname(abspath(__file__)))
from freesas.align import AlignModels

usage = "./supcomb file1.pdb file2.pdb [OPTIONS]"
description = "align several models and calculate NSD"
parser = argparse.ArgumentParser(usage=usage, description=description)
parser.add_argument("file", metavar="FILE", nargs='+', help="pdb files to align")
parser.add_argument("-m", "--mode",dest="mode", type=str, choices=["SLOW", "FAST"], default="SLOW", help="Either SLOW or FAST, default: %(default)s)")
parser.add_argument("-e", "--enantiomorphs",type=str, choices=["YES", "NO"], default="YES", help="Search enantiomorphs, YES or NO, default: %(default)s)")

args = parser.parse_args()

align = AlignModels()

if args.mode=="SLOW":
    align.slow = True
else:
    align.slow = False
if args.enantiomorphs=="YES":
    align.enantiomorphs = True
else:
    align.enantiomorphs = False

output = []
for i in range(len(args.file)):
    if i<9:
        output.append("aligned-0%s.pdb"%(i+1))
    else:
        output.append("aligned-%s.pdb"%(i+1))
print output
align.inputfiles = args.file
align.outputfiles = output
align.assign_models()

if len(args.file)==2:
    dist = align.alignment_2models()
    print "%s and %s aligned"%(args.file[0], args.file[1])
    print "NSD after optimized alignment = %s"%(dist)
else:
    tableNSD = align.makeNSDarray()
    align.alignment_reference()
    print "%s models aligned on the model %s"%(len(args.file), align.reference)