#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: FreeSAS
#             https://github.com/kif/freesas
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#

"""
Contains helper functions for loading SAS data from differents sources.
"""
__authors__ = ["Martha Brennich"]
__contact__ = "martha.brennich@googlemail.com"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/07/2020"
__status__ = "development"
__docformat__ = 'restructuredtext'

from typing import List, Union
from os import PathLike
from numpy import loadtxt, array, ndarray

PathType = Union[PathLike, str, bytes]

def load_scattering_data(filename: PathType) -> ndarray:
    """
    Load scattering data q, I, err into a numpy array.

    :param filename: ASCII file, 3 column (q,I,err)
    :return: numpy array with 3 column (q,I,err)
    """
    try:
        data = loadtxt(filename)
    except OSError:
        raise OSError("File could not be read.")
    except ValueError:
        try:
            with open(filename) as data_file:
                text = data_file.readlines()
        except OSError:
            raise OSError("File could not be read.")
        else:
            try:
                data = parse_ascii_data(text, number_of_columns=3)
            except ValueError:
                raise ValueError("File does not seem to be "
                                 "in the format q, I, err. ")
    return data

def parse_ascii_data(input_file_text: List[str],
                     number_of_columns: int) -> ndarray:
    """
    Parse data from an ascii file into an N column numpy array

    :param input_file_text: List containing one line of input data per element
    :param number_of_columns: Expected number of lines in the data file
    :return: numpy array with 3 column (q,I,err)
    """
    data = []
    for line in input_file_text:
        split = line.split()
        if len(split) == number_of_columns:
            try:
                data.append([float(x) for x in split])
            except ValueError as err:
                if "could not convert string to float" in err.args[0]:
                    pass
                else:
                    raise
    if data == []:
        raise ValueError
    data = array(data)
    return data

def convert_inverse_angstrom_to_nanometer(data_in_inverse_angstrom: ndarray) \
                                            -> ndarray:
    """
    Convert data with q in 1/Å to 1/nm.

    :param data_in_inverse_angstrom: numpy array in format
                                     (q_in_inverse_Angstrom,I,err)
    :return: numpy array with 3 column (q_in_inverse_nm,I,err)
    """
    q_in_angstrom, intensity, err = data_in_inverse_angstrom.T
    return array([q_in_angstrom*10.0, intensity, err]).T
