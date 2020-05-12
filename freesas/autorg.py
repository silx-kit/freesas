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
from ._autorg import RG_RESULT, autoRg, AutoRG, linear_fit, FIT_RESULT, autorg_instance
from scipy.optimize import curve_fit


def auto_gpa(data, Rg_min=1.0, qRg_max=1.5):
    """Uses the GPA theory to guess quickly the 
    radius of gyration and the forwards scattering for a sample
    
    The theory is described in `Guinier peak analysis for visual and automated
    inspection of small-angle X-ray scattering data`
    Christopher D. Putnam
    J. Appl. Cryst. (2016). 49, 1412–1419
    
    This fits sqrt(q²Rg²)*exp(-q²Rg²/3)*I0/Rg to the curve I*q = f(q²)
    
    :param data: the raw data read from disc. Only q and I are used.
    :param Rg_min: the minimal accpetable value for the radius of gyration
    :param qRg_max: the upper bound for the Guinier region. More relaxed as for actual Guinier for 
    :return: autRg result with limited information
    """
    q = data.T[0]
    I = data.T[1]
    start = numpy.argmax(I)
    q = q[start:]
    I = I[start:]

    w = q * q
    f = lambda w, Rg, I0: I0 / Rg * numpy.sqrt(w * Rg ** 2) * numpy.exp(-w * Rg ** 2 / 3)
    res = curve_fit(f, w, q * I, [Rg_min, I[0]])
    rg, I0 = res[0]
    # Reduce the investigation region:
    mask = q < qRg_max / rg
    q = q[mask]
    I = I[mask]
    res = curve_fit(f, q * q, q * I, [rg, I0])
    rg, I0 = res[0]
    sigma_rg, sigma_I0 = numpy.sqrt(numpy.diag(res[1]))
    end = numpy.where(data.T[0] > qRg_max / rg)[0][0]
    return RG_RESULT(rg, sigma_rg, I0, sigma_I0, start, end, -1, 0)

