__author__ = "Guillaume Bonamis"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import argparse
from test.profile_alignment import alignment_with_reference

parser = argparse.ArgumentParser(description="align several model and calculate NSD")
parser.add_argument("file1", help="first model, reference")
parser.add_argument("file2", help="second model, alin with first one")
args = parser.parse_args()
if args.file1 and args.file2:
    models = (args.file1, args.file2)
    pdbout = ("model_aligned-01", "model_aligned-02")
    alignment_with_reference(models, pdbout)
    print "Done"
    