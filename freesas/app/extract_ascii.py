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
__date__ = "25/05/2020"

import os
import sys
import freesas
import argparse
import logging
import glob
import platform
from pyFAI.io import Nexus
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_ascii")

if sys.version_info[0] < 3:
    logger.error("This code requires Python 3.4+")


def parse():

    """ Parse input and return list of files.
    :return: list of input files
    """
    usage = "extract-ascii.py [OPTIONS] FILES "
    description = "Extract the SAXS data from a Nexus files as a 3 column ascii (q, I, err). Metadata are exported in the headers as needed."
    epilog = """extract_ascii.py allows you to export the data in inverse nm or inverse A with possible intensity scaling.   
    """
    version = "extract_ascii.py version %s from %s" % (freesas.version, freesas.date)
    parser = argparse.ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("file", metavar="FILE", nargs='+', help="dat files to compare")
    parser.add_argument("-o", "--output", action='store', help="Output filename, by default the same with .dat extension", default=None, type=str)
    parser.add_argument("-u", "--unit", action='store', help="Unit for q: inverse nm or Angstrom?", default="nm", type=str)
    parser.add_argument("-n", "--normalize", action='store', help="Re-normalize all intensities with this factor ", default=1.0, type=float)
    parser.add_argument("-a", "--all", action='store_true', help="extract every individual frame", default=False)
    parser.add_argument("-v", "--verbose", default=False, help="switch to verbose mode", action='store_true')
    parser.add_argument("-V", "--version", action='version', version=version)
    return parser.parse_args()


def read_nexus_simple(filename):
    "return some NexusJuice from a HDF5 file "
    results = {}
    with Nexus(filename, "r") as nxsr:
        entry_grp = nxsr.get_entries()[0]
        h5path = entry_grp.name
        nxdata_grp = nxsr.h5[entry_grp.attrs["default"]]
        signal = nxdata_grp.attrs["signal"]
        axis = nxdata_grp.attrs["axes"]
        I = nxdata_grp[signal][()]
        q = nxdata_grp[axis][()]
        std = nxdata_grp["error"][()]
        npt = len(q)
        unit = pyFAI.units.to_unit(axis + "_" + nxdata_grp[axis].attrs["units"])
        integration_grp = nxdata_grp.parent
        poni = str(integration_grp["configuration/file_name"][()]).strip()
        if not os.path.exists(poni):
            poni = str(integration_grp["configuration/data"][()]).strip()
        polarization = integration_grp["configuration/polarization_factor"][()]
        method = IntegrationMethod.select_method(**json.loads(integration_grp["configuration/integration_method"][()]))[0]
        instrument_grp = nxsr.get_class(entry_grp, class_type="NXinstrument")[0]
        detector_grp = nxsr.get_class(instrument_grp, class_type="NXdetector")[0]
        mask = detector_grp["pixel_mask"].attrs["filename"]
        mono_grp = nxsr.get_class(instrument_grp, class_type="NXmonochromator")[0]
        energy = mono_grp["energy"][()]
        sample_grp = nxsr.get_class(entry_grp, class_type="NXsample")[0]
        buffer = sample_grp["buffer"]
        concentration = sample_grp["concentration"]
    return NexusJuice(filename, h5path, npt, unit, q, I, poni, mask, energy, polarization, buffer, concentration)


def extract_average(filename):
    "Extract data and metadata from a BM29-Nexus file coming from the averaging step (integratemultiframe plugin)"
    pass


def extract_sub(filename):
    "Extract data and metadata from a BM29-Nexus file coming from the magic subtraction (subtractbuffer plugin)"
    pass


def write_ascii(res, output="output.dat", hdr="#", linesep=os.linesep):
        """
        :param res: named tuple of numpy array containing Scattering vector, Intensity and deviation
        :param outputCurve: name of the 3-column ascii file to be written
        @param hdr: header mark, usually '#'
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
        if self.sample.comments is not None:
            headers.append(hdr + " " + self.sample.comments.value)
        else:
            headers.append(hdr)
        if self.sample.concentration is not None:
            headers.append(hdr + " Sample c= %s mg/ml" % self.sample.concentration.value)
        else:
            headers.append(hdr + " Sample c= -1  mg/ml")
        headers += [hdr, hdr + " Sample environment:"]
        if self.experimentSetup.detector is not None:
            headers.append(hdr + " Detector = %s" % self.experimentSetup.detector.value)
        if self.experimentSetup.pixelSize_1 is not None:
            headers.append(hdr + " PixelSize_1 = %s" % self.experimentSetup.pixelSize_1.value)
        if self.experimentSetup.pixelSize_2 is not None:
            headers.append(hdr + " PixelSize_2 = %s" % self.experimentSetup.pixelSize_2.value)
        headers.append(hdr)
        if self.sample.comments is not None:
            headers.append(hdr + " title = %s" % self.sample.comments.value)
        if (self.experimentSetup.frameNumber is not None) and\
           (self.experimentSetup.frameMax is not None):
            headers.append(hdr + " Frame %s of %s" % (self.experimentSetup.frameNumber.value, self.experimentSetup.frameMax.value))
        if self.experimentSetup.exposureTime is not None:
            headers.append(hdr + " Time per frame (s) = %s" % self.experimentSetup.exposureTime.value)
        if self.experimentSetup.detectorDistance is not None:
            headers.append(hdr + " SampleDistance = %s" % self.experimentSetup.detectorDistance.value)
        if self.experimentSetup.wavelength is not None:
            headers.append(hdr + " WaveLength = %s" % self.experimentSetup.wavelength.value)
        if self.experimentSetup.normalizationFactor is not None:
            headers.append(hdr + " Normalization = %s" % self.experimentSetup.normalizationFactor.value)
        if self.experimentSetup.beamStopDiode is not None:
            headers.append(hdr + " DiodeCurr = %s" % self.experimentSetup.beamStopDiode.value)
        if self.experimentSetup.machineCurrent is not None:
            headers.append(hdr + " MachCurr = %s" % self.experimentSetup.machineCurrent.value)
        if self.experimentSetup.maskFile is not None:
            headers.append(hdr + " Mask = %s" % self.experimentSetup.maskFile.path.value)
        headers.append(hdr)
        headers.append(hdr + " N 3")
        if self.sample.comments is not None:
            headers.append(hdr + " L q*nm  I_%s  stddev" % self.sample.comments.value)
        else:
            headers.append(hdr + " L q*nm  I_  stddev")
        headers.append(hdr)
        headers.append(hdr + " Sample Information:")
        if self.experimentSetup.storageTemperature is not None:
            headers.append(hdr + " Storage Temperature (degrees C): %s" % self.experimentSetup.storageTemperature.value)
        if self.experimentSetup.exposureTemperature is not None:
            headers.append(hdr + " Measurement Temperature (degrees C): %s" % self.experimentSetup.exposureTemperature.value)

        if self.sample.concentration is not None:
            headers.append(hdr + " Concentration: %s" % self.sample.concentration.value)
        else:
            headers.append(hdr + " Concentration: -1")
        if self.sample.code is not None:
            headers.append(hdr + " Code: %s" % self.sample.code.value)
        else:
            headers.append(hdr + " Code: ")

        with open(outputCurve, "w") as f:
            f.writelines(linesep.join(headers))
            f.write(linesep)

            if res.sigma is None:
                data = ["%14.6e %14.6e " % (q, I)
                        for q, I in zip(res.radial, res.intensity)]
                        # 3if abs(I - self.dummy) > self.delta_dummy]
            else:
                data = ["%14.6e %14.6e %14.6e" % (q, I, std)
                        for q, I, std in zip(res.radial, res.intensity, res.sigma)]
                        # if abs(I - self.dummy) > self.delta_dummy]
            data.append("")
            f.writelines(linesep.join(data))
            f.flush()


def main():
    args = parse()
    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    files = [i for i in args.file if os.path.exists(i)]
    if platform.system() == "Windows" and files == []:
        files = glob.glob(args.file[0])
        files.sort()
    input_len = len(files)
    logger.debug("%s input files" % input_len)


if __name__ == "__main__":
    main()
