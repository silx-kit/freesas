#!/usr/bin/env python3
# coding: utf-8
"""Is the poor accuracy of the Shannon Dmax a property of the method or of my
choice of its free parameters?

The paper leaves three knobs under-specified for blind use:
  * nmax, the number of Shannon channels retained (7 to 11 in the paper, but
    derived here from the Kratky cutoff, which gives 9 to 25);
  * the width of the mean filter applied to f1(Dmax) and f2(Dmax);
  * the parameters (m, alpha, qbar0) of the q-adaptive filter on I(q).

This script sweeps each of them on a fixed set of ground-truth cases, so that
the method is not condemned for a bad setting.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

import itertools

import numpy

from synthetic import make_curve
from nsdmax import dmax_shannon

CASES = [
    ("sphere", {}, 8.0),
    ("sphere", {}, 14.0),
    ("spheroid", {"epsilon": 2.0}, 10.0),
    ("spheroid", {"epsilon": 0.5}, 10.0),
    ("core_shell", {"epsilon": 1.5, "X": 0.16}, 10.0),
    ("parabola", {}, 12.0),
]
NOISES = [0.0, 0.05]


def curves():
    from freesas.autorg import auto_guinier

    out = []
    for shape, kwargs, Dmax in CASES:
        for noise in NOISES:
            c = make_curve(shape=shape, Dmax=Dmax, qmin=0.05, qmax=4.0, npt=1024,
                           rel_error_at_qmax=noise, seed=0, **kwargs)
            data = numpy.vstack((c.q, c.I, c.sigma)).T
            g = auto_guinier(data)
            out.append((c, g))
    return out


def score(cases, **kwargs):
    """Median and p90 of |relative error| over the case set."""
    errs = []
    nconv = 0
    for c, g in cases:
        try:
            r = dmax_shannon(c.q, c.I, g.Rg, g.I0, **kwargs)
        except Exception:
            continue
        errs.append(abs(r.Dmax - c.Dmax) / c.Dmax)
        nconv += bool(r.converged)
    if not errs:
        return float("nan"), float("nan"), 0
    return numpy.median(errs), numpy.percentile(errs, 90), nconv


def main():
    cases = curves()
    n = len(cases)

    print(f"{n} ground-truth cases; baseline = Kratky-derived nmax\n")
    med, p90, nc = score(cases)
    print(f"baseline                    median {100*med:6.2f}%  p90 {100*p90:6.2f}%  conv {nc}/{n}")

    print("\n--- forcing nmax (the paper uses 7 to 11) ---")
    for nmax in (4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20):
        med, p90, nc = score(cases, nmax=nmax)
        print(f"nmax = {nmax:3d}                   median {100*med:6.2f}%  p90 {100*p90:6.2f}%  conv {nc}/{n}")

    print("\n--- width of the mean filter on f1/f2 (grid step = Rg/500) ---")
    for widths in ((1,), (11,), (51,), (101,), (201,), (401,),
                   (1, 3, 5, 9, 15, 21, 31), (21, 51, 101, 201),
                   (51, 101, 201, 401)):
        med, p90, nc = score(cases, smoothing_levels=widths)
        tag = str(widths) if len(widths) < 5 else f"{widths[0]}..{widths[-1]} ({len(widths)})"
        print(f"smoothing {tag:24s} median {100*med:6.2f}%  p90 {100*p90:6.2f}%  conv {nc}/{n}")

    print("\n--- q-adaptive filter on I(q) ---")
    for m, alpha, off in itertools.product((5, 20, 50), (1.0, 3.0), (0.3, 0.5, 0.7)):
        med, p90, nc = score(cases, filter_width=m, filter_alpha=alpha,
                             filter_offset=off)
        print(f"m={m:3d} alpha={alpha:4.1f} qbar0={off:4.2f}     median {100*med:6.2f}%"
              f"  p90 {100*p90:6.2f}%  conv {nc}/{n}")

    print("\n--- scan window, in units of Rg ---")
    for window in ((2.5, 3.5), (2.0, 4.0), (1.5, 5.0), (2.8, 3.2), (2.9, 3.1)):
        med, p90, nc = score(cases, dmax_over_rg=window)
        print(f"window {str(window):12s}         median {100*med:6.2f}%  p90 {100*p90:6.2f}%  conv {nc}/{n}")

    print("\n--- best combination found: nmax forced + heavy smoothing ---")
    best = None
    for nmax in (6, 7, 8, 9, 10, 11):
        for widths in ((51,), (101,), (201,), (21, 51, 101, 201)):
            med, p90, nc = score(cases, nmax=nmax, smoothing_levels=widths)
            if best is None or med < best[0]:
                best = (med, p90, nc, nmax, widths)
    med, p90, nc, nmax, widths = best
    print(f"nmax={nmax} smoothing={widths}: median {100*med:6.2f}%  p90 {100*p90:6.2f}%  conv {nc}/{n}")

    print("\n--- conditioning: sensitivity of Dmax to a 1% change in the Rg anchor ---")
    print(f"{'case':32s} {'Rg':>7s} {'D(0.99Rg)':>10s} {'D(Rg)':>8s} {'D(1.01Rg)':>10s} {'spread':>8s}")
    for c, g in cases:
        vals = [dmax_shannon(c.q, c.I, f * g.Rg, g.I0).Dmax for f in (0.99, 1.0, 1.01)]
        spread = (max(vals) - min(vals)) / c.Dmax
        print(f"{c.label[:32]:32s} {g.Rg:7.3f} {vals[0]:10.2f} {vals[1]:8.2f} "
              f"{vals[2]:10.2f} {100*spread:7.1f}%")


if __name__ == "__main__":
    main()
