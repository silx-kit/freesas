__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2017, ESRF"
__date__ = "06/02/2026"

import sys
import os
import zipfile
import posixpath
import logging
from typing import Union
from silx.io.nxdata import NXdata
from dataclasses import dataclass
import numpy

logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError:
    logger.error("H5py is mandatory to parse HDF5 files")
    h5py = None


@dataclass
class IntegratedPattern:
    """Store one pyFAI integrated pattern"""

    point: Union[float, int, None]
    radial: numpy.ndarray
    intensity: numpy.ndarray
    intensity_errors: Union[numpy.ndarray, None] = None
    radial_name: str = ""
    radial_units: str = ""
    intensity_name: str = ""
    intensity_units: str = ""

    def __repr__(self):
        line = f"# {self.radial_name}({self.radial_units}) \t {self.intensity_name}({self.intensity_units})"
        if self.intensity_errors is not None:
            line += " \t uncertainties"
        res = [line]
        if self.intensity_errors is None:
            for q, i, s in zip(self.radial, self.intensity):
                res.append(f"{q} \t {i}")
        else:
            for q, i, s in zip(self.radial, self.intensity, self.intensity_errors):
                res.append(f"{q} \t {i} \t {s}")
        return os.linesep.join(res)


def read_nexus_integrated_patterns(group):
    """Read integrated patterns from a HDF5 NXdata group.

    It reads from both single (1D signal) or multi (2D signal) NXdata.
    :param group : h5py.Group
    :return: list of IntegratedPattern instances.
    """
    nxdata = NXdata(group)
    if not nxdata.is_valid:
        raise RuntimeError(
            f"Cannot parse NXdata group: {group.file.filename}::{group.name}"
        )
    if not (nxdata.signal_is_1d or nxdata.signal_is_2d):
        raise RuntimeError(
            f"Signal is not a 1D or 2D dataset: {group.file.filename}::{group.name}"
        )

    if nxdata.signal_is_1d:
        points = [None]
    else:  # 2d
        if nxdata.axes[0] is None:
            points = [None] * nxdata.signal.shape[0]
        else:
            points = nxdata.axes[0][()]

    if nxdata.axes[-1] is None:
        radial = numpy.arange(nxdata.signal.shape[1])
        radial_units = ""
        radial_name = ""
    else:
        axis_dataset = nxdata.axes[-1]
        radial = axis_dataset[()]
        radial_name = axis_dataset.name.split("/")[-1]
        radial_units = axis_dataset.attrs.get("units", "")

    intensities = numpy.atleast_2d(nxdata.signal)
    intensity_name = nxdata.signal.name.split("/")[-1]
    intensity_units = nxdata.signal.attrs.get("units", "")

    if nxdata.errors is None:
        errors = [None] * intensities.shape[0]
    else:
        errors = numpy.atleast_2d(nxdata.errors)

    if (len(points), len(radial)) != intensities.shape:
        raise RuntimeError("Shape mismatch between axes and signal")

    return [
        IntegratedPattern(
            point,
            radial,
            intensity,
            intensity_errors,
            radial_name,
            radial_units,
            intensity_name,
            intensity_units,
        )
        for point, intensity, intensity_errors in zip(points, intensities, errors)
    ]


class Tree:
    def __init__(self, root=None):
        self.root = root or {}
        self.skip = set()

    def visit_item(self, name, obj):
        if name in self.skip:
            return
        node = self.root
        path = [i.replace(" ", "_") for i in name.split("/")]
        for i in path[:-1]:
            if i not in node:
                node[i] = {}
            node = node[i]
        if isinstance(obj, h5py.Group):
            if obj.attrs.get("NX_class") == "NXdata" and "errors" in obj:
                try:
                    node[path[-1]] = read_nexus_integrated_patterns(obj)
                except (KeyError, OSError) as err:
                    print(f"{type(err).__name__}: {err} while readding {path}")
                for key in obj:
                    self.skip.add(posixpath.join(name, key))
                    if isinstance(obj[key], h5py.Group):
                        for sub in obj[key]:
                            self.skip.add(posixpath.join(name, key, sub))
            else:
                node[path[-1]] = {}
        if isinstance(obj, h5py.Dataset):
            if len(obj.shape) <= 1:
                node[path[-1]] = obj[()]

    def save(self, filename):
        with zipfile.ZipFile(filename, "w") as z:

            def write(path, name, obj):
                new_path = posixpath.join(path, name)
                if isinstance(obj, dict):
                    if sys.version_info >= (3, 11):
                        z.mkdir(new_path)
                    for key, value in obj.items():
                        write(new_path, key, value)
                elif isinstance(obj, numpy.ndarray):
                    if obj.ndim == 1:
                        z.writestr(new_path, os.linesep.join(str(i) for i in obj))
                    else:
                        z.writestr(new_path, str(obj))
                elif isinstance(obj, list):
                    if sys.version_info >= (3, 11):
                        z.mkdir(new_path)
                    if len(obj) == 1:
                        fname = new_path + "/biosaxs.dat"
                        z.writestr(fname, str(obj[0]))
                    else:
                        for i, j in enumerate(obj):
                            fname = new_path + f"/bioxaxs_{i:04d}.dat"
                            z.writestr(fname, str(j))
                elif isinstance(obj, (int, float, numpy.number, bool, numpy.bool)):
                    z.writestr(new_path, str(obj))
                elif isinstance(obj, (str, bytes)):
                    z.writestr(new_path, obj)
                else:
                    print(f"skip {new_path} for {obj} of type {obj.__class__.__mro__}")

            root = self.root
            for key, value in root.items():
                write("", key, value)

    def get(self, path):
        node = self.root
        for i in path.split("/"):
            node = node[i]
        return node


def convert_nexus2zip(nexusfile, outfile=None):
    """Convert a nexus-file, as produced by BM29 beamline into a zip file

    :param nexusfile: string with the path of the input file
    :param outfile: name of the output file, unless, just replace the extension with ".zip"
    :return: nothing, maybe an error code ?
    """
    tree = Tree()
    with h5py.File(nexusfile, "r") as h:
        h.visititems(tree.visit_item)
    outfile = outfile or (os.path.splitext(nexusfile)[0] + ".h5")
    tree.save(outfile)
