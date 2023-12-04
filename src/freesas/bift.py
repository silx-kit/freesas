# -*- coding: utf-8 -*-
"""
Bayesian Inverse Fourier Transform

This code is the implementation of
Steen Hansen J. Appl. Cryst. (2000). 33, 1415-1421

Based on the BIFT from Jesse Hopkins, available at:
https://sourceforge.net/p/bioxtasraw/git/ci/master/tree/bioxtasraw/BIFT.py

Many thanks to Pierre Paleo for the auto-alpha guess
"""

__authors__ = ["Jerome Kieffer", "Jesse Hopkins"]
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "10/06/2020"

import logging
logger = logging.getLogger(__name__)
# from collections import namedtuple
from math import log, ceil
import numpy
from scipy.optimize import minimize
from ._bift import BIFT
from .autorg import auto_gpa, autoRg, auto_guinier, NoGuinierRegionError


def auto_bift(data, Dmax=None, alpha=None, npt=100,
              start_point=None, end_point=None,
              scan_size=11, Dmax_over_Rg=3):
    """Calculates the inverse Fourier tranform of the data using an optimisation of the evidence

    :param data: 2D array with q, I(q), δI(q). q can be in 1/nm or 1/A, it imposes the unit for r & Dmax
    :param Dmax: Maximum diameter of the object, this is the starting point to be refined. Can be guessed
    :param alpha: Regularisation parameter, let it to None for automatic scan
    :param npt: Number of point for the curve p(r)
    :param start_point: First useable point in the I(q) curve, this is not the start of the Guinier region
    :param end_point: Last useable point in the I(q) curve
    :param scan_size: size of the initial geometrical scan for alpha values.
    :param Dmax_over_Rg: In average, protein's Dmax is 3x Rg, use this to adjust
    :return: BIFT object. Call the get_best to retrieve the optimal solution
    """
    assert data.ndim == 2
    assert data.shape[1] == 3  # enforce q, I, err
    use_wisdom = False
    data = data[slice(start_point, end_point)]
    q, I, err = data.T
    npt = min(npt, q.size)  # no chance for oversampling !
    bo = BIFT(q, I, err)  # this is the bift object
    if Dmax is None:
        # Try to get a reasonable guess from Rg
        try:
            Guinier = auto_guinier(data)
        except:
            logger.error("Guinier analysis failed !")
            raise
#         print(Guinier)
        if Guinier.Rg <= 0:
            raise NoGuinierRegionError
        Dmax = bo.set_Guinier(Guinier, Dmax_over_Rg)
    if alpha is None:
        alpha_max = bo.guess_alpha_max(npt)
        # First scan on alpha:
        key = bo.grid_scan(Dmax, Dmax, 1,
                           1.0 / alpha_max, alpha_max, scan_size, npt)
        Dmax, alpha = key[:2]
        # Then scan on Dmax:
        key = bo.grid_scan(max(Dmax / 2, Dmax * (Dmax_over_Rg - 1) / Dmax_over_Rg), Dmax * (Dmax_over_Rg + 1) / Dmax_over_Rg, scan_size,
                           alpha, alpha, 1, npt)
        Dmax, alpha = key[:2]
        if bo.evidence_cache[key].converged:
            bo.update_wisdom()
            use_wisdom = True

    # Optimization using Bayesian operator:
    logger.info("Start search at Dmax=%.2f alpha=%.2f use wisdom=%s", Dmax, alpha, use_wisdom)
    res = minimize(bo.opti_evidence, (Dmax, log(alpha)), args=(npt, use_wisdom), method="powell")
    logger.info("Result of optimisation:\n  %s", res)
    return bo
