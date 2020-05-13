# -*- coding: utf-8 -*-
"""
Functions to generating graphs related to 
"""

__authors__ = ["Jerome Kieffer"]
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "13/05/2020"

import logging
logger = logging.getLogger(__name__)
import numpy
from ._autorg import RG_RESULT, autoRg, AutoRG, linear_fit, FIT_RESULT, autorg_instance
from scipy.optimize import curve_fit


def auto_gpa(data, Rg_min=1.0, qRg_max=1.5):
    """Uses the GPA theory to guess quickly Rg, the 
    radius of gyration and I0, the forwards scattering
    
    The theory is described in `Guinier peak analysis for visual and automated
    inspection of small-angle X-ray scattering data`
    Christopher D. Putnam
    J. Appl. Cryst. (2016). 49, 1412–1419
    
    This fits sqrt(q²Rg²)*exp(-q²Rg²/3)*I0/Rg to the curve I*q = f(q²)
    
    The Guinier region goes arbitrary from 0.75 to 1.5 q·Rg 
    qRg_max can be provided 
    
    :param data: the raw data read from disc. Only q and I are used.
    :param Rg_min: the minimal accpetable value for the radius of gyration
    :param qRg_max: the upper bound for the Guinier region. More relaxed as for actual Guinier for 
    :return: autRg result with limited information
    """
    q = data.T[0]
    I = data.T[1]
    start = numpy.argmax(I)
    stop = numpy.where(q > qRg_max / Rg_min)[0][0]
    q = q[start:stop]
    I = I[start:stop]

    x = q * q
    y = I * q
    p1 = numpy.argmax(y)

    # Those are guess from the max position:
    Rg = (1.5 / x[p1]) ** 0.5
    I0 = I[p1] * numpy.exp(x[p1] * Rg ** 2 / 3)

    # Let's cut-down the guinier region from 0.75-1.5 in qRg
    try:
        start = numpy.where(q > qRg_max / Rg / 2.0)[0][0]
    except IndexError:
        start = None
    try:
        stop = numpy.where(q > qRg_max / Rg)[0][0]
    except IndexError:
        stop = None
    q = q[start:stop]
    I = I[start:stop]

    x = q * q
    y = I * q

    f = lambda w, Rg, I0: I0 / Rg * numpy.sqrt(w * Rg * Rg) * numpy.exp(-w * Rg * Rg / 3)
    res = curve_fit(f, x, y, [Rg, I0])
    logger.debug("GPA upgrade Rg %s-> %s and I0 %s -> %s", Rg, res[0][0], I0, res[0][1])
    Rg, I0 = res[0]
    sigma_Rg, sigma_I0 = numpy.sqrt(numpy.diag(res[1]))
    end = numpy.where(data.T[0] > qRg_max / Rg)[0][0]
    start = numpy.where(data.T[0] > qRg_max / Rg / 2.0)[0][0]
    return RG_RESULT(Rg, sigma_Rg, I0, sigma_I0, start, end, -1, 0)

