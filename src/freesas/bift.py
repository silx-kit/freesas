"""
Bayesian Inverse Fourier Transform

This code is the implementation of
Steen Hansen J. Appl. Cryst. (2000). 33, 1415-1421

Based on the BIFT from Jesse Hopkins, available at:
https://sourceforge.net/p/bioxtasraw/git/ci/master/tree/bioxtasraw/BIFT.py

Many thanks to Pierre Paleo for the auto-alpha guess
"""

__authors__ = ["Jérôme Kieffer", "Jesse Hopkins"]
__license__ = "MIT"
__copyright__ = "2020-2026, ESRF"
__date__ = "06/02/2026"

import logging
from math import log

from scipy.optimize import minimize

from ._bift import BIFT
from .autorg import NoGuinierRegionError, auto_guinier

logger = logging.getLogger(__name__)


def auto_bift(
    data,
    Dmax=None,
    alpha=None,
    npt=100,
    start_point=None,
    end_point=None,
    scan_size=11,
    Dmax_over_Rg=3,
    fit_background=False,
):
    """Calculates the inverse Fourier tranform of the data using an optimisation of the evidence

    :param data: 2D array with q, I(q), δI(q). q can be in 1/nm or 1/A, it imposes the unit for r & Dmax
    :param Dmax: Maximum diameter of the object, this is the starting point to be refined. Can be guessed
    :param alpha: Regularisation parameter, let it to None for automatic scan
    :param npt: Number of point for the curve p(r)
    :param start_point: First useable point in the I(q) curve, this is not the start of the Guinier region
    :param end_point: Last useable point in the I(q) curve
    :param scan_size: size of the initial geometrical scan for alpha values.
    :param Dmax_over_Rg: In average, protein's Dmax is 3x Rg, use this to adjust
    :param fit_background: adjust a flat background B0 together with p(r). Recommended when
                           the data extend to large q, where a residual incoherent background
                           dominates the form factor. B0 is degenerate with a p(r) peaked at
                           r->0, so it is only well constrained when q_max*Dmax/npt >> 1.
                           Being a nuisance parameter, B0 also soaks up any model error
                           (finite npt, truncated p(r)): its absolute value is not the
                           physical background, only differences between datasets are.
    :return: BIFT object. Call the get_best to retrieve the optimal solution
    """
    assert data.ndim == 2
    assert data.shape[1] == 3  # enforce q, I, err
    use_wisdom = False
    data = data[slice(start_point, end_point)]
    q, intenisty, err = data.T
    npt = min(npt, q.size)  # no chance for oversampling !
    bo = BIFT(q, intenisty, err, fit_background)  # this is the bift object
    if Dmax is None:
        # Try to get a reasonable guess from Rg
        try:
            Guinier = auto_guinier(data)
        except Exception:
            logger.error("Guinier analysis failed !")
            raise
        #         print(Guinier)
        if Guinier.Rg <= 0:
            raise NoGuinierRegionError
        Dmax = bo.set_Guinier(Guinier, Dmax_over_Rg)
    if alpha is None:
        alpha_max = bo.guess_alpha_max(npt)
        # First scan on alpha:
        key = bo.grid_scan(Dmax, Dmax, 1, 1.0 / alpha_max, alpha_max, scan_size, npt)
        Dmax, alpha = key[:2]
        # Then scan on Dmax:
        key = bo.grid_scan(
            max(Dmax / 2, Dmax * (Dmax_over_Rg - 1) / Dmax_over_Rg),
            Dmax * (Dmax_over_Rg + 1) / Dmax_over_Rg,
            scan_size,
            alpha,
            alpha,
            1,
            npt,
        )
        Dmax, alpha = key[:2]
        if bo.evidence_cache[key].converged:
            bo.update_wisdom()
            use_wisdom = True

    # Optimization using Bayesian operator:
    logger.info(
        "Start search at Dmax=%.2f alpha=%.2f use wisdom=%s", Dmax, alpha, use_wisdom
    )
    res = minimize(
        bo.opti_evidence, (Dmax, log(alpha)), args=(npt, use_wisdom), method="powell"
    )
    logger.info("Result of optimisation:\n  %s", res)
    return bo
