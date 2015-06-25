#!/usr/bin/python
__author__ = "Guillaume Bonamis"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import argparse
from os.path import dirname, abspath
base = dirname(dirname(abspath(__file__)))
from freesas.align import AlignModels
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("log_freesas")

usage = "./supcomb FILES [OPTIONS]"
description = "align several models and calculate NSD"
parser = argparse.ArgumentParser(usage=usage, description=description)
parser.add_argument("file", metavar="FILE", nargs='+', help="pdb files to align")
parser.add_argument("-m", "--mode",dest="mode", type=str, choices=["SLOW", "FAST"], default="SLOW", help="Either SLOW or FAST, default: %(default)s)")
parser.add_argument("-e", "--enantiomorphs",type=str, choices=["YES", "NO"], default="YES", help="Search enantiomorphs, YES or NO, default: %(default)s)")
parser.add_argument("-q", "--quiet", type=str, choices=["ON", "OFF"], default="ON", help="Hide log or not, default: %(default)s")
parser.add_argument("-o", "--output", type=str, default="aligned.pdb", help="output filename, default: %(default)s")

args = parser.parse_args()
input_len = len(args.file)
logger.info("%s input files"%input_len)
align = AlignModels()

if args.mode=="SLOW":
    align.slow = True
    logger.info("SLOW mode")
else:
    align.slow = False
    logger.info("FAST mode")

if args.enantiomorphs=="YES":
    align.enantiomorphs = True
else:
    align.enantiomorphs = False
    logger.info("NO enantiomorphs")

if args.quiet=="OFF":
    logger.setLevel(logging.DEBUG)
    logger.info("setLevel: Debug")

if input_len==2:
    align.outputfiles = args.output
else:
    output = []
    for i in range(input_len):
        if i<9:
            output.append("aligned-0%s.pdb"%(i+1))
        else:
            output.append("aligned-%s.pdb"%(i+1))
    align.outputfiles = output

align.inputfiles = args.file
align.assign_models()

if input_len==2:
    dist = align.alignment_2models()
    logger.info("%s and %s aligned"%(args.file[0], args.file[1]))
    logger.info("NSD after optimized alignment = %s"%(dist))
else:
    align.makeNSDarray()
    align.alignment_reference()
    logger.info("%s models aligned on the model %s"%(input_len, align.reference+1))
    align.plotNSDarray()