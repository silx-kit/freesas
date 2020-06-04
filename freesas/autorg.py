# -*- coding: utf-8 -*-
"""
Functions to generating graphs related to 
"""

__authors__ = ["Jerome Kieffer"]
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "03/06/2020"

import logging
logger = logging.getLogger(__name__)
import math
import numpy
from ._autorg import RG_RESULT, autoRg, AutoGuinier, linear_fit, FIT_RESULT, guinier, NoGuinierRegionError, DTYPE, InsufficientDataError
from scipy.optimize import curve_fit


def auto_gpa(data, Rg_min=1.0, qRg_max=1.3, qRg_min=0.5):
    """Uses the GPA theory to guess quickly Rg, the 
    radius of gyration and I0, the forwards scattering
    
    The theory is described in `Guinier peak analysis for visual and automated
    inspection of small-angle X-ray scattering data`
    Christopher D. Putnam
    J. Appl. Cryst. (2016). 49, 1412–1419
    
    This fits sqrt(q²Rg²)*exp(-q²Rg²/3)*I0/Rg to the curve I*q = f(q²)
    
    The Guinier region goes arbitrary from 0.5 to 1.3 q·Rg 
    qRg_min and qRg_max can be provided 
    
    :param data: the raw data read from disc. Only q and I are used.
    :param Rg_min: the minimal accpetable value for the radius of gyration
    :param qRg_max: the default upper bound for the Guinier region.
    :param qRg_min: the default lower bound for the Guinier region. 
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

    # Let's cut-down the guinier region from 0.5-1.3 in qRg
    try:
        start = numpy.where(q > qRg_min / Rg)[0][0]
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
    start = numpy.where(data.T[0] > qRg_min / Rg)[0][0]
    return RG_RESULT(Rg, sigma_Rg, I0, sigma_I0, start, end, -1, 0)


def auto_guinier(data, Rg_min=1.0, qRg_max=1.3, relax=1.2):
    """
    Yet another implementation of the Guinier fit
    
    The idea:
    * extract the reasonable range
    * convert to the Guinier space (ln(I) = f(q²)
    * scan all possible intervall
    * keep any with qRg_max<1.3 (or 1.5 in relaxed mode)
    * select the begining and the end of the guinier region according to the contribution of two parameters:
      - (q_max·Rg - q_min·Rg)/qRg_max --> in favor of large ranges
      - 1 / RMSD                      --> in favor of good quality data 
      For each start and end point, the contribution of all ranges are averaged out (using histograms)
      The best solution is the start/end position with the maximum average.
    * All ranges within this region are averaged out to measure Rg, I0 and more importantly their deviation.   
    * The quality is still to be calculated
    * Aggergation is assessed according a second order polynom fit. 
     
    :param data: 2D array with (q,I,err)
    :param Rg_min: minimum value for Rg
    :param qRg_max: upper bound of the Guinier region
    :param relax: relaxation factor for the upper bound
    :param resolution: step size of the slope histogram
    :return: autRg result 
    """

    raw_size = data.shape[0]
    q_ary = numpy.empty(raw_size, dtype=DTYPE)
    i_ary = numpy.empty(raw_size, dtype=DTYPE)
    sigma_ary = numpy.empty(raw_size, dtype=DTYPE)
    q2_ary = numpy.empty(raw_size, dtype=DTYPE)
    lnI_ary = numpy.empty(raw_size, dtype=DTYPE)
    wg_ary = numpy.empty(raw_size, dtype=DTYPE)

    start0, stop0 = guinier.currate_data(data, q_ary, i_ary, sigma_ary,
                                         Rg_min, qRg_max, relax)
    if start0 < 0:
        raise InsufficientDataError("Minimum region size is %s" % guinier.min_size)
    guinier.guinier_space(start0, stop0, q_ary, i_ary, sigma_ary,
                                         q2_ary, lnI_ary, wg_ary)

    fits = guinier.many_fit(q2_ary, lnI_ary, wg_ary, start0, stop0, Rg_min, qRg_max, relax)

    cnt, relaxed, qRg_max, aslope_max = guinier.count_valid(fits, qRg_max, relax)
    # valid_fits = fits[fits[:, 9] < qRg_max]
    if cnt == 0:
        raise NoGuinierRegionError(qRg_max)

#     npt = math.ceil(aslope_max / resolution) + 1
#     distribution = guinier.slope_distribution(fits, npt, resolution, qRg_max)
#
#     best = numpy.argmax(distribution)
#     if best >= 1 and best < npt - 1:
#         gradient = 0.5 * (distribution[best + 1] - distribution[best - 1])
#         hessian = distribution[best + 1] + distribution[best - 1] - 2 * distribution[best]
#         if hessian == 0:
#             best_corr = 0
#         else:
#             best_corr = -gradient / hessian
#             if abs(best_corr) > 1:
#                 best_corr = 0
#     aslope = resolution * (best + best_corr)
#
#     drg = abs(aslope + valid_fits[:, 10])
#     # drg2 = drg * drg
#     # w2 = (valid_fits[:, 19] - valid_fits[:, 18]) / qRg_max ** 2
#     w = (valid_fits[:, 9] - valid_fits[:, 8]) / qRg_max
#     solution = numpy.argmin(drg / w)
#
#     start, stop = valid_fits[solution][4:6]
    # select the Guinier region based on all fits:
    start, stop = guinier.find_region(fits, qRg_max)

    # Now average out the
    Rg_avg, Rg_std, I0_avg, I0_std, good = guinier.average_values(fits, start, stop)

    aggregated = guinier.check_aggregation(q2_ary, lnI_ary, wg_ary, start0, stop, threshold=None)

    quality = (Rg_avg / Rg_std) * (good / cnt)
    if relaxed:
        quality *= 0.6
    if aggregated:
        quality *= 0.6
    result = RG_RESULT(Rg_avg, Rg_std, I0_avg, I0_std, start, stop, numpy.clip(quality, 0, 1), aggregated)
    return result
