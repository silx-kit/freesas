#!/usr/bin/python3
# coding: utf-8
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2020  European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "15/01/2021"

import io
import os
import sys
import logging
import glob
import platform
import posixpath
from collections import namedtuple, OrderedDict
import json
import copy
import pyFAI
from pyFAI.io import Nexus
from .sas_argparser import SASParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_ascii")

if sys.version_info[0] < 3:
    logger.error("This code requires Python 3.4+")

NexusJuice = namedtuple("NexusJuice", "filename h5path npt unit q I poni mask energy polarization signal2d error2d buffer concentration")


def parse():

    """ Parse input and return list of files.
    :return: list of input files
    """
    description = "Extract the SAXS data from a Nexus files as a 3 column ascii (q, I, err). Metadata are exported in the headers as needed."
    epilog = """extract_ascii.py allows you to export the data in inverse nm or inverse A with possible intensity scaling.
    """
    parser = SASParser(prog="extract-ascii.py", description=description, epilog=epilog)
    # Commented option need to be implemented
    # parser.add_argument("-o", "--output", action='store', help="Output filename, by default the same with .dat extension", default=None, type=str)
    # parser.add_argument("-u", "--unit", action='store', help="Unit for q: inverse nm or Angstrom?", default="nm", type=str)
    # parser.add_argument("-n", "--normalize", action='store', help="Re-normalize all intensities with this factor ", default=1.0, type=float)
    parser.add_file_argument("HDF5 input data")
    parser.add_argument("-a", "--all", action='store_true', help="extract every individual frame", default=False)
    return parser.parse_args()


def extract_averaged(filename):
    "return some infomations extracted from a HDF5 file "
    results = OrderedDict()
    results["filename"] = filename
    # Missing: comment normalization
    with Nexus(filename, "r") as nxsr:
        entry_grp = nxsr.get_entries()[0]
        results["h5path"] = entry_grp.name
        nxdata_grp = nxsr.h5[entry_grp.attrs["default"]]
        signal = nxdata_grp.attrs["signal"]
        axis = nxdata_grp.attrs["axes"]
        results["I"] = nxdata_grp[signal][()]
        results["q"] = nxdata_grp[axis][()]
        results["std"] = nxdata_grp["errors"][()]
        results["unit"] = pyFAI.units.to_unit(axis + "_" + nxdata_grp[axis].attrs["units"])
        integration_grp = nxdata_grp.parent
        results["geometry"] = json.loads(integration_grp["configuration/data"][()])
        results["polarization"] = integration_grp["configuration/polarization_factor"][()]

        instrument_grps = nxsr.get_class(entry_grp, class_type="NXinstrument")
        if instrument_grps:
            detector_grp = nxsr.get_class(instrument_grps[0], class_type="NXdetector")[0]
            results["mask"] = detector_grp["pixel_mask"].attrs["filename"]
        sample_grp = nxsr.get_class(entry_grp, class_type="NXsample")[0]
        results["sample"] = posixpath.split(sample_grp.name)[-1]
        results["buffer"] = sample_grp["buffer"][()]
        results["storage temperature"] = sample_grp["temperature_env"][()]
        results["exposure temperature"] = sample_grp["temperature"][()]
        results["concentration"] = sample_grp["concentration"][()]
        if "2_correlation_mapping" in entry_grp:
            results["to_merge"] = entry_grp["2_correlation_mapping/results/to_merge"][()]
    return results


def extract_all(filename):
    "return some infomations extracted from a HDF5 file for  all individual frames"
    res = []
    results = OrderedDict()
    results["filename"] = filename
    with Nexus(filename, "r") as nxsr:
        entry_grp = nxsr.get_entries()[0]
        results["h5path"] = entry_grp.name
        nxdata_grp = nxsr.h5[entry_grp.name + "/1_integration/results"]
        signal = nxdata_grp.attrs["signal"]
        axis = nxdata_grp.attrs["axes"][1]
        I = nxdata_grp[signal][()]
        results["q"] = nxdata_grp[axis][()]
        std = nxdata_grp["errors"][()]
        results["unit"] = pyFAI.units.to_unit(axis + "_" + nxdata_grp[axis].attrs["units"])
        integration_grp = nxdata_grp.parent
        results["geometry"] = json.loads(integration_grp["configuration/data"][()])
        results["polarization"] = integration_grp["configuration/polarization_factor"][()]
        instrument_grp = nxsr.get_class(entry_grp, class_type="NXinstrument")[0]
        detector_grp = nxsr.get_class(instrument_grp, class_type="NXdetector")[0]
        results["mask"] = detector_grp["pixel_mask"].attrs["filename"]
        sample_grp = nxsr.get_class(entry_grp, class_type="NXsample")[0]
        results["sample"] = posixpath.split(sample_grp.name)[-1]
        results["buffer"] = sample_grp["buffer"][()]
        if "temperature_env" in sample_grp:
            results["storage temperature"] = sample_grp["temperature_env"][()]
        if "temperature" in sample_grp:
            results["exposure temperature"] = sample_grp["temperature"][()]
        if "concentration" in sample_grp:
            results["concentration"] = sample_grp["concentration"][()]
#         if "2_correlation_mapping" in entry_grp:
#             results["to_merge"] = entry_grp["2_correlation_mapping/results/to_merge"][()]
    for i, s in zip(I, std):
        r = copy.copy(results)
        r["I"] = i
        r["std"] = s
        res.append(r)
    return res


def write_ascii(results, output=None, hdr="#", linesep=os.linesep):
    """
    :param resusts: dict containing some NexusJuice
    :param output: name of the 3-column ascii file to be written
    :param hdr: header mark, usually '#'
    :param linesep: to be able to addapt the end of lines

Adam Round explicitelly asked for (email from Date: Tue, 04 Oct 2011 15:22:29 +0200) :
Modification from:
# BSA buffer
# Sample c= 0.0 mg/ml (these two lines are required for current DOS pipeline and can be cleaned up once we use EDNA to get to ab-initio models)
#
# Sample environment:
# Detector = Pilatus 1M
# PixelSize_1 = 0.000172
# PixelSize_2 = 6.283185 (I think it could avoid confusion if we give teh actual pixel size as 0.000172 for X and Y and not to give the integrated sizes. Also could there also be a modification for PixelSize_1 as on the diagonal wont it be the hypotenuse (0.000243)? and thus will be on average a bit bigger than 0.000172)
#
# title = BSA buffer
# Frame 7 of 10
# Time per frame (s) = 10
# SampleDistance = 2.43
# WaveLength = 9.31e-11
# Normalization = 0.0004885
# History-1 = saxs_angle +pass -omod n -rsys normal -da 360_deg -odim = 1 /data/id14eh3/inhouse/saxs_pilatus/Adam/EDNAtests/2d/dumdum_008_07.edf/data/id14eh3/inhouse/saxs_pilatus/Adam/EDNAtests/misc/dumdum_008_07.ang
# DiodeCurr = 0.0001592934
# MachCurr = 163.3938
# Mask = /data/id14eh3/archive/CALIBRATION/MASK/Pcon_01Jun_msk.edf
# SaxsDataVersion = 2.40
#
# N 3
# L q*nm  I_BSA buffer  stddev
#
# Sample Information:
# Storage Temperature (degrees C): 4
# Measurement Temperature (degrees C): 10
# Concentration: 0.0
# Code: BSA
s-vector Intensity Error
s-vector Intensity Error
s-vector Intensity Error
s-vector Intensity Error
        """
    hdr = str(hdr)
    headers = []
    if "comments" in results:
        headers.append(hdr + " " + results["comments"])
    else:
        headers.append(hdr)
    headers.append(hdr + " Sample c= %s mg/ml" % results.get("concentration", -1))
    headers += [hdr, hdr + " Sample environment:"]
    if "geometry" in results:
        headers.append(hdr + " Detector = %s" % results["geometry"]["detector"])
        headers.append(hdr + " SampleDistance = %s" % results["geometry"]["dist"])
        headers.append(hdr + " WaveLength = %s" % results["geometry"]["wavelength"])
    headers.append(hdr)
    if "comments" in results:
        headers.append(hdr + " title = %s" % results["comment"])
    if "to_merge" in results:
        headers.append(hdr + " Frames merged: " + " ".join([str(i) for i in results["to_merge"]]))
    if 'normalization' in results:
        headers.append(hdr + " Normalization = %s" % results["normalization"])
    if "mask" in results:
        headers.append(hdr + " Mask = %s" % results["mask"])
    headers.append(hdr)
    headers.append(hdr + (" N 3" if "std" in results else " N 2"))
    line = hdr + " L "
    if "unit" in results:
        a, b = str(results["unit"]).split("_")
        line += a + "*" + b.strip("^-1") + "  I_"
    else:
        line += "q  I_"
    if "comment" in results:
        line += results["comments"]
    if "std" in results:
        line += "  stddev"
    headers.append(line)
    headers.append(hdr)
    headers.append(hdr + " Sample Information:")
    if "storage temperature" in results:
        headers.append(hdr + " Storage Temperature (degrees C): %s" % results["storage temperature"])
    if "exposure temperature" in results:
        headers.append(hdr + " Measurement Temperature (degrees C): %s" % results["exposure temperature"])

    headers.append(hdr + " Concentration: %s" % results.get("concentration", -1))
    if "buffer" in results:
        headers.append(hdr + " Buffer: %s" % results["buffer"])
    headers.append(hdr + " Code: %s" % results.get("sample", ""))

    def write(headers, file_):

        file_.writelines(linesep.join(headers))
        file_.write(linesep)

        if "std" in results:
            data = ["%14.6e\t%14.6e\t%14.6e" % (q, I, std)
                    for q, I, std in zip(results["q"], results["I"], results["std"])]
        else:
            data = ["%14.6e\t%14.6e\t" % (q, I)
                    for q, I in zip(results["q"], results["I"])]
        data.append("")
        file_.writelines(linesep.join(data))

    if output:
        with open(output, "w") as f:
            write(headers, f)
    else:
        f = io.StringIO()
        write(headers, f)
        f.seek(0)
        return f.read()


def main():
    args = parse()
    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    files = [i for i in args.file if os.path.exists(i)]
    if platform.system() == "Windows" and files == []:
        files = glob.glob(args.file[0])
        files.sort()
    input_len = len(files)
    logger.debug("%s input files", input_len)
    for src in files:
        if args.all:
            dest = os.path.splitext(src)[0] + "%04i.dat"
            for idx, frame in enumerate(extract_all(src)):
                print(src, " --> ", dest % idx)
                write_ascii(frame, dest % idx)
        else:
            dest = os.path.splitext(src)[0] + ".dat"
            write_ascii(extract_averaged(src), dest)
            print(src, " --> ", dest)


if __name__ == "__main__":
    main()
