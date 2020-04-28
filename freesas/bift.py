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
__date__ = "27/04/2020"

import logging
logger = logging.getLogger(__name__)
# from collections import namedtuple
from math import log
import numpy
from scipy.optimize import minimize
from ._bift import BIFT
from .autorg import autoRg
from .decorators import timeit


def auto_bift(data, Dmax=None, alpha=None, npt=100,
              start_point=None, end_point=None, scan_size=21, Dmax_over_Rg=3):
    """Calculates the inverse Fourier tranform of the data using an optimisation of the evidence 
    
    :param data: 2D array with q, I(q), δI(q). q can be in 1/nm or 1/A, it imposes the unit for r & Dmax
    :param Dmax: Maximum diameter of the object, this is the starting point to be refined. Can be guessed
    :param alpha: Regularisation parameter, let it to None for automatic scan
    :param npt: Number of point for the curve p(r)
    :param start_point: First useable point in the I(q) curve
    :param end_point: Last useable point in the I(q) curve
    :param scan_size: size of the initial geometrical scan for alpha values.
    :param Dmax_over_Rg: In average, protein's Dmax is 3x Rg, use this to adjust
    :return: BIFT object. Call the get_best to retrieve the optimal solution  
    """
    assert data.ndim == 2
    assert data.shape[1] == 3  # enforce q, I, err
    data = data[slice(start_point, end_point)]
    q, I, err = data.T
    npt = min(npt, q.size)  # no chance for oversampling !
    bo = BIFT(q, I, err)  # this is the bift object
    if Dmax is None:
        # Try to get a reasonable from Rg
        rg = autoRg(data)
        Dmax = bo.set_Guinier(rg, Dmax_over_Rg)
    if alpha is None:
        alpha_max = bo.guess_alpha_max(npt)
        alpha = bo.grid_scan(Dmax, Dmax, 1, 1.0 / alpha_max, alpha_max, scan_size, npt)[1]

    # Optimization using Bayesian operator:
    logger.info("Start search at alpha=%.2f Dmax=%.2f", alpha, Dmax)
    res = minimize(bo.opti_evidence, (Dmax, log(alpha)), args=(npt,), method="powell")
    logger.info("Result of optimisation:\n  %s", res)
    return bo


def extrapolate_q(ift, q):
    """This probvides a curve I=f(q) with an extrapolated q-range to zero
    
    :param ift: an BIFT instance with the best Dmax/alpha couple found.
    :param
    """
    pass


if __name__ == "__main__":
    import sys
    data = numpy.loadtxt(sys.argv[1])

