# -*- coding: utf-8 -*-
"""
Functions to generating graphs related to 
"""

__authors__ = ["Jerome Kieffer"]
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "12/05/2020"

import logging
logger = logging.getLogger(__name__)
import numpy

from collections import namedtuple

def auto_gpa(data, Rg_min=1.0, qRg_max=1.5):
    """Uses the GPA theory to guess quickly the 
    radius of gyration and the forwards scattering for a sample
    
    The theory is described in `Guinier peak analysis for visual and automated
    inspection of small-angle X-ray scattering data`
    Christopher D. Putnam
    J. Appl. Cryst. (2016). 49, 1412–1419
    
    :param data: the raw data read from disc. Only q and I are used.
    :param Rg_min: the minimal accpetable value for the radius of gyration
    :param qRg_max: the upper bound for the Guinier region
    :Return 
