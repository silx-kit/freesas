#!/usr/bin/env python3
# coding: utf-8
"""A plateau-onset ("knee") estimator built on the same Shannon formalism.

Measured behaviour of the residuals of De Caro et al. (2026):

    Rg_NS(Dmax) and I0_NS(Dmax) SATURATE. Under-estimating Dmax aliases the
    alternating series and the moments come out wrong; over-estimating it is
    harmless -- extra channels are added and the series still converges to the
    same moments. So f1(Dmax) and f2(Dmax) are not V-shaped: they are high
    below the true Dmax and flat at the noise floor above it.

Looking for the *minimum* of a plateau is ill-posed (the argmin wanders over
the whole plateau, and a 1% error on the Guinier anchor moves it by tens of
percent). Looking for the *onset* of the plateau is well-posed, and it does
not depend on the absolute residual level -- which is what the accuracy of the
Guinier anchor sets.

This module implements that estimator and compares it with the published one.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

from collections import namedtuple

import numpy
from scipy.ndimage import uniform_filter1d

from nsdmax import dmax_residuals, q_adaptive_filter

KNEE_RESULT = namedtuple("KNEE_RESULT", "Dmax plateau threshold converged")


def _KNEE_RESULT_repr(self):
    flag = "" if self.converged else " NOT-CONVERGED"
    return (f"Dmax={self.Dmax:6.3f} plateau={100*self.plateau:5.2f}% "
            f"thr={100*self.threshold:5.2f}%{flag}")


KNEE_RESULT.__repr__ = _KNEE_RESULT_repr


def dmax_knee(
    q,
    I,
    Rg_gui,
    I0_gui,
    dmax_over_rg=(1.2, 6.0),
    step=None,
    filter_width=20,
    nmax=None,
    power=-4.0,
    factor=1.5,
    hold=0.15,
    smooth=None,
    full_output=False,
):
    """Estimate Dmax as the onset of the plateau of max(f1, f2).

    :param q: scattering vector
    :param I: scattering intensity
    :param Rg_gui, I0_gui: anchors from the Guinier fit
    :param dmax_over_rg: scan bounds, in units of Rg. Wide on purpose: the
                         estimator must not depend on a narrow prior window.
    :param step: scan step, defaults to Rg/200
    :param nmax: number of Shannon channels, None -> from the Kratky cutoff
    :param factor: the plateau is considered reached below factor x its level
    :param hold: the residual must stay below threshold over this fraction of
                 the scan range, to reject isolated dips
    :param smooth: width of the mean filter on the residuals, in points
    :return: KNEE_RESULT
    """
    q = numpy.ascontiguousarray(q, dtype=numpy.float64)
    I = numpy.ascontiguousarray(I, dtype=numpy.float64)
    I_filtered = q_adaptive_filter(q, I, m=filter_width)

    if step is None:
        step = Rg_gui / 200.0
    lo, hi = (f * Rg_gui for f in dmax_over_rg)
    Dmax_array = numpy.arange(lo, hi + step, step)

    f1, f2, nmax_array = dmax_residuals(
        q, I_filtered, Dmax_array, Rg_gui, I0_gui, nmax=nmax, power=power
    )
    residual = numpy.maximum(f1, f2)
    residual[~numpy.isfinite(residual)] = numpy.nanmax(residual[numpy.isfinite(residual)])
    if smooth is None:
        smooth = max(3, Dmax_array.size // 40)
    residual = uniform_filter1d(residual, int(smooth), mode="nearest")

    # plateau level: the residual floor, taken on the upper half of the scan
    upper = residual[residual.size // 2:]
    plateau = float(numpy.median(upper))
    threshold = factor * plateau

    # first index from which the residual stays below threshold for `hold`
    span = max(1, int(hold * residual.size))
    below = residual < threshold
    index = None
    for i in range(residual.size - span):
        if below[i:i + span].all():
            index = i
            break
    converged = index is not None
    if not converged:
        index = int(numpy.argmin(residual))

    result = KNEE_RESULT(
        Dmax=float(Dmax_array[index]),
        plateau=plateau,
        threshold=threshold,
        converged=converged,
    )
    if full_output:
        return result, {"Dmax_array": Dmax_array, "residual": residual,
                        "f1": f1, "f2": f2, "nmax_array": nmax_array}
    return result


def main():
    """Compare the published estimator and the knee, on ground truth."""
    from freesas.autorg import auto_guinier
    from freesas.sasio import load_scattering_data
    from synthetic import make_curve
    from nsdmax import dmax_shannon

    print("=== apoferritin SASDFN8, published Dmax = 12.64 nm ===")
    data = load_scattering_data("/home/kieffer/.claude/jobs/0356cb97/tmp/SASDFN8.dat")
    q, I, s = data.T
    g = auto_guinier(data)
    pub = dmax_shannon(q, I, g.Rg, g.I0, dmax_over_rg=(2.0, 3.5))
    kne = dmax_knee(q, I, g.Rg, g.I0)
    print(f"  Guinier Rg = {g.Rg:.3f} nm (paper 5.237)")
    print(f"  published estimator : {pub.Dmax:6.3f} nm  ({100*(pub.Dmax/12.64-1):+6.2f}%)")
    print(f"  knee estimator      : {kne.Dmax:6.3f} nm  ({100*(kne.Dmax/12.64-1):+6.2f}%)  {kne}")

    print("\n=== synthetic ground truth ===")
    cases = [("sphere", {}, 8.0), ("sphere", {}, 14.0),
             ("spheroid", {"epsilon": 2.0}, 10.0),
             ("spheroid", {"epsilon": 0.5}, 10.0),
             ("core_shell", {"epsilon": 1.5, "X": 0.16}, 10.0),
             ("parabola", {}, 12.0)]
    print(f"  {'case':34s} {'true':>6s} {'3Rg':>8s} {'published':>10s} {'knee':>8s} {'conv':>6s}")
    err_pub, err_knee, err_3rg = [], [], []
    for noise in (0.0, 0.05, 0.10):
        for shape, kw, D in cases:
            c = make_curve(shape=shape, Dmax=D, qmin=0.05, qmax=4.0, npt=1024,
                           rel_error_at_qmax=noise, seed=0, **kw)
            gg = auto_guinier(numpy.vstack((c.q, c.I, c.sigma)).T)
            p = dmax_shannon(c.q, c.I, gg.Rg, gg.I0)
            k = dmax_knee(c.q, c.I, gg.Rg, gg.I0)
            err_pub.append(abs(p.Dmax - D) / D)
            err_knee.append(abs(k.Dmax - D) / D)
            err_3rg.append(abs(3 * gg.Rg - D) / D)
            print(f"  {c.label[:34]:34s} {D:6.1f} {3*gg.Rg:8.2f} {p.Dmax:10.2f} "
                  f"{k.Dmax:8.2f} {str(k.converged):>6s}")
    print(f"\n  median |err|:  3Rg {100*numpy.median(err_3rg):5.2f}%   "
          f"published {100*numpy.median(err_pub):5.2f}%   "
          f"knee {100*numpy.median(err_knee):5.2f}%")

    print("\n=== conditioning: +/-2% on the Guinier Rg anchor ===")
    print(f"  {'case':34s} {'published spread':>18s} {'knee spread':>13s}")
    for shape, kw, D in cases:
        c = make_curve(shape=shape, Dmax=D, qmin=0.05, qmax=4.0, npt=1024,
                       rel_error_at_qmax=0.05, seed=0, **kw)
        gg = auto_guinier(numpy.vstack((c.q, c.I, c.sigma)).T)
        pv = [dmax_shannon(c.q, c.I, f * gg.Rg, gg.I0).Dmax for f in (0.98, 1.0, 1.02)]
        kv = [dmax_knee(c.q, c.I, f * gg.Rg, gg.I0).Dmax for f in (0.98, 1.0, 1.02)]
        print(f"  {c.label[:34]:34s} {100*(max(pv)-min(pv))/D:17.1f}% "
              f"{100*(max(kv)-min(kv))/D:12.1f}%")


if __name__ == "__main__":
    main()
