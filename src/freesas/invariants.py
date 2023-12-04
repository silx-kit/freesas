# -*- coding: utf-8 -*-
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
"""
This module is mainly about the calculation of the Rambo-Tainer invariant
described in:

https://dx.doi.org/10.1038%2Fnature12070  

Some formula taken from Putnam et al, 2007, Table 1 in the review
"""
__authors__ = ["Martha E. Brennich", "J. Kieffer"]
__license__ = "MIT"
__date__ = "10/06/2020"

import logging
logger = logging.getLogger(__name__)
import numpy
from .collections import RT_RESULT


def extrapolate(data, guinier):
    """Extrapolate SAS data according to the Guinier fit until q=0
    Uncertainties are extrapolated (linearly) from the Guinier region 
    
    :param data: SAS data in q,I,dI format
    :param guinier: result of a Guinier fit
    :return: extrapolated SAS data 
    """
    
    dq = data[1, 0] - data[0, 0]
    qmin = data[guinier.start_point, 0]

    q_low = numpy.arange(0, qmin, dq)
    # Extrapolate I from Guinier approximation: 
    I_low = guinier.I0 * numpy.exp(-(q_low**2 * guinier.Rg**2) / 3.0)
    # Extrapolate dI from Guinier region:
    range_ = slice(guinier.start_point, guinier.end_point+1)
    slope, intercept = numpy.polyfit(data[range_, 0], data[range_, 2], deg=1)
    dI_low = abs(q_low*slope + intercept)
    # Now wrap-up
    data_low = numpy.vstack((q_low, I_low, dI_low)).T
    return numpy.concatenate((data_low, data[guinier.start_point:]))


def calc_Porod(data, guinier):
    """Calculate the particle volume according to Porod's formula:
    
    V = 2*π²I₀²/(sum_q I(q)q² dq)
    
    Formula from Putnam's review, 2007, table 1
    Intensities are extrapolated to q=0 using Guinier fit.
    
    :param data:  SAS data in q,I,dI format
    :param Guinier: result of a Guinier fit (instance of RT_RESULT)
    :return: Volume calculated according to Porrod's formula
    """ 
    q, I, dI = extrapolate(data, guinier).T
    
    denom = numpy.trapz(I*q**2, q)
    volume = 2*numpy.pi**2*guinier.I0 / denom
    return volume


def calc_Vc(data, Rg, dRg, I0, dI0, imin):
    """Calculates the Rambo-Tainer invariant Vc, including extrapolation to q=0
    
    :param data:  SAS data in q,I,dI format, cropped to maximal q that should be used for calculation (normally 2 nm-1)
    :param Rg,dRg,I0,dI0:  results from Guinier approximation/autorg
    :param imin:  minimal index of the Guinier range, below that index data will be extrapolated by the Guinier approximation
    :returns: Vc and an error estimate based on non-correlated error propagation
    """
    dq = data[1, 0] - data[0, 0]
    qmin = data[imin, 0]
    qlow = numpy.arange(0, qmin, dq)

    lowqint = numpy.trapz((qlow * I0 * numpy.exp(-(qlow * qlow * Rg * Rg) / 3.0)), qlow)
    dlowqint = numpy.trapz(qlow * numpy.sqrt((numpy.exp(-(qlow * qlow * Rg * Rg) / 3.0) * dI0) ** 2 + ((I0 * 2.0 * (qlow * qlow) * Rg / 3.0) * numpy.exp(-(qlow * qlow * Rg * Rg) / 3.0) * dRg) ** 2), qlow)
    vabs = numpy.trapz(data[imin:, 0] * data[imin:, 1], data[imin:, 0])
    dvabs = numpy.trapz(data[imin:, 0] * data[imin:, 2], data[imin:, 0])
    vc = I0 / (lowqint + vabs)
    dvc = (dI0 / I0 + (dlowqint + dvabs) / (lowqint + vabs)) * vc
    return (vc, dvc)


def calc_Rambo_Tainer(data,
                      guinier, qmax=2.0):
    """calculates the invariants Vc and Qr from the Rambo & Tainer 2013 Paper,
    also the the mass estimate based on Qr for proteins
    
    :param data: data in q,I,dI format, q in nm^-1
    :param guinier: RG_RESULT instance with result from the Guinier fit
    :param qmax: maximum q-value for the calculation in nm^-1
    @return: dict with Vc, Qr and mass plus errors
    """
    scale_prot = 1.0 / 0.1231
    power_prot = 1.0

    imax = abs(data[:, 0] - qmax).argmin()
    if (imax <= guinier.start_point) or (guinier.start_point < 0):  # unlikely but can happened
        logger.error("Guinier region start too late for Rambo_Tainer invariants calculation")
        return None
    vc = calc_Vc(data[:imax, :], guinier.Rg, guinier.sigma_Rg, guinier.I0, guinier.sigma_I0, guinier.start_point)

    qr = vc[0] ** 2 / (guinier.Rg)
    mass = scale_prot * qr ** power_prot

    dqr = qr * (guinier.sigma_Rg / guinier.Rg + 2 * ((vc[1]) / (vc[0])))
    dmass = mass * dqr / qr

    return RT_RESULT(vc[0], vc[1], qr, dqr, mass, dmass)
