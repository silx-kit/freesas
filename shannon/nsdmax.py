#!/usr/bin/env python3
# coding: utf-8
"""Determination of Dmax from a SAXS profile via Nyquist-Shannon sampling.

Prototype implementation of:

    De Caro, Alberga, Scattarella, Sibillano, Mangini, Zhou, Stoll & Giannini
    "Shannon sampling based approach for the structural solution of nano-objects
    by laboratory and synchrotron SAXS data"
    J. Appl. Cryst. (2026). 59, 1170-1183 + supporting information

Equation numbers refer to the paper (n) and its supporting information (Sn).

This is evaluation code: it lives outside src/freesas on purpose and depends
only on numpy/scipy so it can be run without building the project.

The unit of Dmax is the inverse of the unit of q (nm if q is in 1/nm), so this
implementation is unit-covariant: feed it 1/A and Dmax comes out in A.

Two limits established by the evaluation (see RAPPORT.md):

  * the first Shannon channel must be measured, i.e. Dmax <= pi/q_min. Beyond
    that the alternating series loses its dominant term and the estimate
    degrades abruptly (section 6.3);
  * the accuracy is set by the Rg anchor, not by the formalism: substituting a
    better Rg halves the error on experimental data (section 6.4).
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

import logging
from collections import namedtuple

import numpy
from numpy import pi
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

DMAX_RESULT = namedtuple(
    "DMAX_RESULT",
    "Dmax sigma_Dmax Dmax_left Dmax_right nmax nsolution converged",
)


def _DMAX_RESULT_repr(self):
    status = "" if self.converged else " NOT-CONVERGED"
    return (
        f"Dmax={self.Dmax:6.3f}(±{self.sigma_Dmax:6.3f}) "
        f"[l={self.Dmax_left:6.3f} r={self.Dmax_right:6.3f}] "
        f"nmax={self.nmax} n_sol={self.nsolution}{status}"
    )


DMAX_RESULT.__repr__ = _DMAX_RESULT_repr


def q_adaptive_filter(q, I, m=20, alpha=1.0, q0_bar=0.5, qmax=None):
    """Blend the raw profile with a mean-filtered one, with a weight growing with q.

    Implements equations (S31) and (S32): the low-q region is left untouched,
    the noisy high-q region is averaged.

        I(q)_f = (1 - w) I(q) + w <I(q)>_m
        w(qbar) = 1 / (1 + ((1-qbar)/qbar * qbar0/(1-qbar0))**alpha)

    :param q: scattering vector, strictly increasing
    :param I: scattering intensity
    :param m: width of the mean filter, in points (S3: 15-30 in the paper)
    :param alpha: non-linearity of the filter. 1 -> almost linear (the paper's
                  best fit for every dataset), >>1 -> sigmoid centred on q0_bar
    :param q0_bar: offset of the filter, in units of qmax
    :param qmax: normalisation of q, defaults to q[-1]
    :return: filtered intensity
    """
    q = numpy.ascontiguousarray(q, dtype=numpy.float64)
    I = numpy.ascontiguousarray(I, dtype=numpy.float64)
    if qmax is None:
        qmax = q[-1]
    qbar = numpy.clip(q / qmax, 1e-12, 1.0 - 1e-12)
    smoothed = uniform_filter1d(I, max(1, int(m)), mode="nearest")
    ratio = (1.0 - qbar) / qbar * q0_bar / (1.0 - q0_bar)
    weight = 1.0 / (1.0 + ratio**alpha)
    return (1.0 - weight) * I + weight * smoothed


def kratky_cutoff(q, I):
    """Estimate q_kr, beyond which noise dominates the I(q).q^2 profile.

    Follows section 2.2 of the paper and Figure S2: q_kr is where the Kratky
    profile crosses its own mean value <I q^2> from below, at high q, i.e. the
    start of the final excursion above the mean.

    :param q: scattering vector
    :param I: scattering intensity
    :return: q_kr
    """
    kratky = I * q**2
    mean = kratky.mean()
    above = kratky > mean
    if not above[-1]:
        # Kratky profile never blows up: all the measured range is usable
        return q[-1]
    # walk back over the trailing run of points above the mean
    idx = len(above) - 1
    while idx > 0 and above[idx - 1]:
        idx -= 1
    return q[idx]


def shannon_sums(q, I, Dmax, nmax, interpolate=False):
    """Sums over the Nyquist-Shannon channels q_n = n.pi/Dmax.

    Vectorised over Dmax. Channel intensities are taken as the value of the
    (smoothed) profile at the measured q closest to n.pi/Dmax.

    :param q: scattering vector, strictly increasing
    :param I: (smoothed) scattering intensity
    :param Dmax: scalar or 1D array of candidate maximum sizes
    :param nmax: scalar or 1D array (matching Dmax) of channel counts
    :param interpolate: linearly interpolate In at q_n instead of taking the
                        nearest measured point. The paper prescribes nearest,
                        but that caps the accuracy of In wherever I(q) varies
                        fast, i.e. exactly at the fringes that carry the size
                        information.
    :return: S, A, I_nmax with
             S = sum (-1)^(n+1) In                  -> I(0) = 2 S     (eq. 3)
             A = sum [1-6/(n.pi)^2] (-1)^(n+1) In   -> Rg  (eq. 5)
             I_nmax = intensity of the last channel (anchor for eq. 10-11)
    """
    Dmax = numpy.atleast_1d(numpy.asarray(Dmax, dtype=numpy.float64))
    nmax = numpy.broadcast_to(numpy.atleast_1d(numpy.asarray(nmax)), Dmax.shape)
    n_top = int(nmax.max())

    n = numpy.arange(1, n_top + 1, dtype=numpy.float64)
    # (n_Dmax, n_top) grid of channel positions
    qn = n[None, :] * pi / Dmax[:, None]
    if interpolate:
        In = numpy.interp(qn, q, I)
    else:
        # nearest measured point: searchsorted then pick the closer neighbour
        idx = numpy.searchsorted(q, qn).clip(1, q.size - 1)
        left_closer = (qn - q[idx - 1]) < (q[idx] - qn)
        idx = numpy.where(left_closer, idx - 1, idx)
        In = I[idx]

    # mask channels beyond nmax (which depends on Dmax) and beyond the data
    valid = (n[None, :] <= nmax[:, None]) & (qn <= q[-1])
    sign = (-1.0) ** (n + 1)
    weight = 1.0 - 6.0 / (n * pi) ** 2

    S = (In * sign[None, :] * valid).sum(axis=1)
    A = (In * (sign * weight)[None, :] * valid).sum(axis=1)
    # intensity of the last valid channel
    last = valid.sum(axis=1) - 1
    I_nmax = numpy.where(last >= 0, In[numpy.arange(In.shape[0]), last.clip(0)], 0.0)
    return S, A, I_nmax


def truncation_corrections(I_nmax, nmax, power=-4.0, terms=1000):
    """Estimate the contribution of the unobserved channels n > nmax.

    Implements equations (10) and (11), extrapolating In ~ I_nmax (n/nmax)^power
    with power = -4 for the Porod law of a sharp-surfaced globular object.

    :param I_nmax: intensity of the last observed channel (scalar or array)
    :param nmax: index of the last observed channel (scalar or array)
    :param power: exponent of the q-scaling of the unobserved intensities
    :param terms: upper limit of the sums (1000 in the paper ~ infinity)
    :return: I_mis, r_mis
    """
    I_nmax = numpy.atleast_1d(numpy.asarray(I_nmax, dtype=numpy.float64))
    nmax = numpy.broadcast_to(
        numpy.atleast_1d(numpy.asarray(nmax, dtype=numpy.float64)), I_nmax.shape
    )
    n = numpy.arange(1, terms + 1, dtype=numpy.float64)
    sign = (-1.0) ** (n + 1)
    weight = 1.0 - 6.0 / (n * pi) ** 2
    # only n > nmax contribute
    beyond = n[None, :] > nmax[:, None]
    ratio = numpy.where(beyond, (n[None, :] / nmax[:, None]) ** power, 0.0)
    I_mis = 2.0 * I_nmax * (ratio * sign[None, :]).sum(axis=1)
    r_mis = I_nmax * (ratio * (sign * weight)[None, :]).sum(axis=1)
    return I_mis, r_mis


def dmax_residuals(q, I, Dmax_array, Rg_gui, I0_gui, nmax=None, power=-4.0,
                   interpolate=False):
    """The two dimensionless residuals f1 and f2 of equations (6) and (7).

    f1 compares the normalised gyration radius Rg/Dmax from the Shannon sums
    with the one from the Guinier fit; f2 does the same on I(0). Both are
    minimal for the true Dmax.

    :param q: scattering vector
    :param I: smoothed scattering intensity
    :param Dmax_array: candidate values of Dmax to probe
    :param Rg_gui: radius of gyration from the Guinier fit
    :param I0_gui: forward intensity from the Guinier fit
    :param nmax: number of Shannon channels. None -> derived per Dmax from the
                 Kratky cutoff; an int -> forced to that value
    :param power: exponent for the truncation corrections
    :return: f1, f2, nmax_array
    """
    Dmax_array = numpy.asarray(Dmax_array, dtype=numpy.float64)
    if nmax is None:
        q_kr = kratky_cutoff(q, I)
        nmax_array = numpy.floor(q_kr * Dmax_array / pi).astype(int).clip(1)
    else:
        nmax_array = numpy.full(Dmax_array.shape, int(nmax))

    S, A, I_nmax = shannon_sums(q, I, Dmax_array, nmax_array, interpolate=interpolate)
    I_mis, r_mis = truncation_corrections(I_nmax, nmax_array, power=power)

    I0_ns = 2.0 * S + I_mis
    num = A + r_mis
    with numpy.errstate(invalid="ignore", divide="ignore"):
        rg_ns = numpy.sqrt(num / I0_ns)  # = Rg_NS / Dmax, eq. 5 + corrections
        rg_gui = Rg_gui / Dmax_array
        f1 = numpy.abs(rg_gui - rg_ns) / rg_gui
        f2 = numpy.abs(I0_gui - I0_ns) / I0_gui
    # unphysical sums (negative ratio) must never win the argmin
    bad = ~numpy.isfinite(f1) | (num <= 0) | (I0_ns <= 0)
    f1[bad] = numpy.inf
    f2[~numpy.isfinite(f2)] = numpy.inf
    return f1, f2, nmax_array


def dmax_shannon(
    q,
    I,
    Rg_gui,
    I0_gui,
    dmax_over_rg=(2.5, 3.5),
    step=None,
    filter_width=20,
    filter_alpha=1.0,
    filter_offset=0.5,
    smoothing_levels=(1, 3, 5, 9, 15, 21, 31),
    tolerance=0.01,
    nmax=None,
    power=-4.0,
    interpolate=False,
    full_output=False,
):
    """Estimate Dmax from a SAXS profile through Nyquist-Shannon sampling.

    Implements section 2.1 of the paper: scan Dmax, locate the minima of f1
    (eq. 6) and f2 (eq. 7), keep only the pairs of minima that agree within
    `tolerance` (eq. 12), average them (eq. 13).

    :param q: scattering vector, strictly increasing
    :param I: scattering intensity, on any scale consistent with I0_gui
    :param Rg_gui: radius of gyration from the Guinier fit
    :param I0_gui: forward intensity from the Guinier fit
    :param dmax_over_rg: bounds of the scan, in units of Rg_gui
    :param step: scan step. None -> Rg_gui/500, i.e. ~0.1 A for a protein
    :param filter_width: m, width of the mean filter in q_adaptive_filter
    :param filter_alpha: alpha of the q-adaptive filter
    :param filter_offset: qbar0 of the q-adaptive filter
    :param smoothing_levels: widths of the mean filter applied to f1 and f2.
                             This is the parameter the paper scans.
    :param tolerance: threshold of equation (12)
    :param nmax: force the number of Shannon channels instead of deriving it
    :param power: exponent for the truncation corrections
    :param full_output: also return the scan, for plotting
    :return: DMAX_RESULT, or (DMAX_RESULT, dict) if full_output
    """
    q = numpy.ascontiguousarray(q, dtype=numpy.float64)
    I = numpy.ascontiguousarray(I, dtype=numpy.float64)
    if Rg_gui <= 0:
        raise ValueError("Rg from the Guinier fit must be positive")

    I_filtered = q_adaptive_filter(
        q, I, m=filter_width, alpha=filter_alpha, q0_bar=filter_offset
    )

    if step is None:
        step = Rg_gui / 500.0
    lo, hi = (f * Rg_gui for f in dmax_over_rg)
    Dmax_array = numpy.arange(lo, hi + step, step)

    f1, f2, nmax_array = dmax_residuals(
        q, I_filtered, Dmax_array, Rg_gui, I0_gui, nmax=nmax, power=power,
        interpolate=interpolate,
    )

    lefts, rights = [], []
    for width in smoothing_levels:
        if width <= 1:
            s1, s2 = f1, f2
        else:
            s1 = uniform_filter1d(f1, int(width), mode="nearest")
            s2 = uniform_filter1d(f2, int(width), mode="nearest")
        d_l = Dmax_array[numpy.argmin(s1)]
        d_r = Dmax_array[numpy.argmin(s2)]
        if abs(d_l - d_r) / (d_l + d_r) < tolerance:  # eq. 12
            lefts.append(d_l)
            rights.append(d_r)

    converged = bool(lefts)
    if converged:
        d_l = float(numpy.mean(lefts))
        d_r = float(numpy.mean(rights))
    else:
        # no smoothing level satisfies eq. 12: report the unfiltered minima so
        # the caller can see how far apart they are, but flag the failure
        d_l = float(Dmax_array[numpy.argmin(f1)])
        d_r = float(Dmax_array[numpy.argmin(f2)])
        logger.info(
            "Equation 12 never satisfied: Dmax_l=%.2f vs Dmax_r=%.2f", d_l, d_r
        )

    result = DMAX_RESULT(
        Dmax=0.5 * (d_l + d_r),  # eq. 13
        sigma_Dmax=0.5 * abs(d_l - d_r),
        Dmax_left=d_l,
        Dmax_right=d_r,
        nmax=int(numpy.floor(numpy.interp(0.5 * (d_l + d_r), Dmax_array, nmax_array))),
        nsolution=len(lefts),
        converged=converged,
    )
    if full_output:
        return result, {
            "Dmax_array": Dmax_array,
            "f1": f1,
            "f2": f2,
            "nmax_array": nmax_array,
            "I_filtered": I_filtered,
        }
    return result
