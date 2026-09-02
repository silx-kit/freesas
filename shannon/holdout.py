#!/usr/bin/env python3
# coding: utf-8
"""Held-out validation and final comparison of the four Dmax estimators.

The x1.15 de-biasing of the knee estimator was calibrated on a first set of
48 synthetic cases. This script freezes that constant and evaluates on an
INDEPENDENT set: different sizes, different shape parameters, different noise
seeds, plus polydisperse and two-population curves that none of the tuning
saw.

Estimators:
  3Rg        the heuristic already inside auto_bift
  published  argmin of f1/f2 over [2.5, 3.5].Rg, De Caro et al. (2026)
  knee       plateau onset of max(f1, f2), same formalism, x1.15 de-biased
  dnn        freesas.dnn, Rg+Dmax.keras
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

import numpy

from synthetic import make_curve, sphere, spheroid, exact_moments, add_noise, CURVE
from nsdmax import dmax_shannon
from knee import dmax_knee

KNEE_CALIBRATION = 1.1495  # frozen, fitted on the tuning set
KNEE_FACTOR = 1.1


def polydisperse_curve(Dmax, spread=0.15, npop=9, qmin=0.05, qmax=4.0, npt=1024,
                       noise=0.05, seed=0):
    """Log-normal-ish size distribution of spheres, Dmax = largest diameter.

    Polydispersity smears the fringes; this is the effect the paper models
    with its q-adaptive filter, so it is a fair and relevant stress test.
    """
    q = numpy.linspace(qmin, qmax, npt)
    sizes = numpy.linspace(1.0 - 2 * spread, 1.0, npop) * Dmax
    # volume-squared weighting, as scattering intensity scales with V^2
    weights = sizes**6
    intensity = numpy.zeros_like(q)
    for size, weight in zip(sizes, weights):
        intensity += weight * sphere(q, size)
    intensity /= weights.sum()
    # exact moments of the mixture, from the low-q expansion
    qlow = numpy.linspace(1e-6, 0.3 / Dmax, 400)
    low = numpy.zeros_like(qlow)
    for size, weight in zip(sizes, weights):
        low += weight * sphere(qlow, size)
    low /= weights.sum()
    coefs = numpy.polyfit(qlow**2, numpy.log(low), 2)
    I0 = float(numpy.exp(coefs[-1]))
    Rg = float(numpy.sqrt(-3.0 * coefs[-2]))
    noisy, sigma = add_noise(q, intensity, noise, seed=seed)
    return CURVE(q, noisy, sigma, Dmax, Rg, I0,
                 f"polydisperse Dmax={Dmax:g} spread={spread:g} noise={noise:g}")


def two_population_curve(Dmax, ratio=0.5, fraction=0.2, qmin=0.05, qmax=4.0,
                         npt=1024, noise=0.05, seed=0):
    """Large species plus a smaller one, Dmax set by the large species."""
    q = numpy.linspace(qmin, qmax, npt)
    big = sphere(q, Dmax)
    small = sphere(q, ratio * Dmax)
    wbig, wsmall = (1.0 - fraction), fraction * ratio**6
    intensity = (wbig * big + wsmall * small) / (wbig + wsmall)
    qlow = numpy.linspace(1e-6, 0.3 / Dmax, 400)
    low = (wbig * sphere(qlow, Dmax) + wsmall * sphere(qlow, ratio * Dmax)) / (wbig + wsmall)
    coefs = numpy.polyfit(qlow**2, numpy.log(low), 2)
    I0 = float(numpy.exp(coefs[-1]))
    Rg = float(numpy.sqrt(-3.0 * coefs[-2]))
    noisy, sigma = add_noise(q, intensity, noise, seed=seed)
    return CURVE(q, noisy, sigma, Dmax, Rg, I0,
                 f"two-pop Dmax={Dmax:g} ratio={ratio:g} f={fraction:g} noise={noise:g}")


def build_holdout():
    """Cases that the calibration never saw."""
    out = []
    # sizes and shape parameters different from the tuning set
    for shape, kw, sizes in (
        ("sphere", {}, (5.0, 7.0, 9.0, 11.0, 15.0)),
        ("spheroid", {"epsilon": 1.3}, (9.0, 13.0)),
        ("spheroid", {"epsilon": 3.0}, (9.0, 13.0)),
        ("spheroid", {"epsilon": 0.35}, (9.0, 13.0)),
        ("core_shell", {"epsilon": 1.2, "X": 0.10}, (9.0, 13.0)),
        ("core_shell", {"epsilon": 1.8, "X": 0.22}, (9.0, 13.0)),
        ("parabola", {}, (7.0, 11.0, 15.0)),
    ):
        for Dmax in sizes:
            for noise, seed in ((0.03, 7), (0.08, 11), (0.15, 13)):
                out.append(make_curve(shape=shape, Dmax=Dmax, qmin=0.05, qmax=4.0,
                                      npt=1024, rel_error_at_qmax=noise, seed=seed,
                                      **kw))
    for Dmax in (8.0, 12.0):
        for spread in (0.10, 0.25):
            out.append(polydisperse_curve(Dmax, spread=spread, noise=0.05, seed=21))
        for fraction in (0.1, 0.3):
            out.append(two_population_curve(Dmax, fraction=fraction, noise=0.05, seed=23))
    return out


def main():
    from freesas.autorg import auto_guinier, auto_gpa
    from freesas.dnn import KerasDNN
    from freesas.resources import resource_filename

    dnn = KerasDNN(resource_filename("keras_models/Rg+Dmax.keras"))
    cases = build_holdout()
    print(f"held-out set: {len(cases)} cases\n")

    results = {"3Rg": [], "published": [], "knee": [], "dnn": []}
    detail = []
    for c in cases:
        data = numpy.vstack((c.q, c.I, c.sigma)).T
        try:
            g = auto_guinier(data)
        except Exception:
            try:
                g = auto_gpa(data)
            except Exception:
                continue
        est = {
            "3Rg": 3.0 * g.Rg,
            "published": dmax_shannon(c.q, c.I, g.Rg, g.I0).Dmax,
            "knee": KNEE_CALIBRATION * dmax_knee(c.q, c.I, g.Rg, g.I0,
                                                 factor=KNEE_FACTOR).Dmax,
            "dnn": float(dnn(c.q, c.I)[1]),
        }
        # every estimator but the network is anchored on the Guinier fit: when
        # that fit fails, the case says nothing about the estimators
        est["_gui_err"] = abs(g.Rg - c.Rg) / c.Rg
        for k, v in est.items():
            if not k.startswith("_"):
                results[k].append(abs(v - c.Dmax) / c.Dmax)
        detail.append((c, est))

    bad = [(c, e) for c, e in detail if e["_gui_err"] > 0.20]
    good = [(c, e) for c, e in detail if e["_gui_err"] <= 0.20]
    print(f"Guinier fit usable in {len(good)}/{len(detail)} cases "
          f"({len(bad)} rejected, |dRg| > 20%)")

    def report(title, subset):
        print(f"\n{title}  (n={len(subset)})")
        print(f"  {'method':11s} {'median|err|':>12s} {'p90|err|':>10s} {'max|err|':>10s}")
        stats = []
        for k in results:
            errs = [abs(e[k] - c.Dmax) / c.Dmax for c, e in subset
                    if numpy.isfinite(e[k])]
            errs = [e for e in errs if numpy.isfinite(e)]
            if errs:
                stats.append((numpy.median(errs), numpy.percentile(errs, 90),
                              max(errs), k))
        for med, p90, mx, k in sorted(stats):
            print(f"  {k:11s} {100*med:11.2f}% {100*p90:9.2f}% {100*mx:9.2f}%")

    report("ALL HELD-OUT CASES (including failed Guinier fits)", detail)
    report("USABLE GUINIER FIT", good)
    report("Dmax <= 10 nm", [(c, e) for c, e in good if c.Dmax <= 10])
    report("Dmax > 10 nm", [(c, e) for c, e in good if c.Dmax > 10])
    report("polydisperse / two-population",
           [(c, e) for c, e in good
            if c.label.startswith(("polydisperse", "two-pop"))])
    report("monodisperse analytical shapes",
           [(c, e) for c, e in good
            if not c.label.startswith(("polydisperse", "two-pop"))])

    print("\n--- worst 8 cases for the knee estimator (usable Guinier only) ---")
    worst = sorted(good, key=lambda ce: -abs(ce[1]["knee"] - ce[0].Dmax) / ce[0].Dmax)
    print(f"  {'case':44s} {'true':>6s} {'knee':>7s} {'dnn':>7s} {'pub':>7s} {'3Rg':>7s}")
    for c, e in worst[:8]:
        print(f"  {c.label[:44]:44s} {c.Dmax:6.1f} {e['knee']:7.2f} {e['dnn']:7.2f} "
              f"{e['published']:7.2f} {e['3Rg']:7.2f}")


if __name__ == "__main__":
    main()
