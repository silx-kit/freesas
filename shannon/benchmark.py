#!/usr/bin/env python3
# coding: utf-8
"""Compare Dmax estimators on synthetic SAXS curves with known ground truth.

Estimators under test:

  shannon      Nyquist-Shannon approach of De Caro et al. (2026), fed with the
               Rg and I(0) of FreeSAS' auto_guinier, as it would be used in
               production.
  shannon_ex   Same, but fed with the *exact* Rg and I(0). Ablation isolating
               the Shannon step from the error of the Guinier fit.
  dnn          The dense neural network of freesas.dnn (Rg+Dmax.keras).
  three_rg     Dmax = 3 Rg_guinier, the heuristic already used by auto_bift.
               The control: an estimator that does not beat it is worthless.
  bift         auto_bift, the current reference. Slow, run on a subset only.

Usage:
    python benchmark.py [--bift] [--out results.csv]
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

import argparse
import csv
import logging
import sys
import time

import numpy

from synthetic import make_curve
from nsdmax import dmax_shannon

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark")

# --- the grid of test cases -------------------------------------------------
SHAPES = [
    ("sphere", {}),
    ("spheroid", {"epsilon": 2.0}),
    ("spheroid", {"epsilon": 0.5}),
    ("core_shell", {"epsilon": 1.5, "X": 0.16}),
    ("parabola", {}),
]
DMAX_VALUES = [4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0]
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]
SEEDS = [0, 1, 2]
QMAX = 4.0  # native grid of the neural network, in 1/nm
QMIN = 0.05
NPT = 1024


def build_estimators(use_bift):
    """Instantiate the estimators, returning a dict name -> callable."""
    from freesas.autorg import auto_guinier, auto_gpa
    from freesas.dnn import KerasDNN
    from freesas.resources import resource_filename

    dnn = KerasDNN(resource_filename("keras_models/Rg+Dmax.keras"))

    def guinier(curve):
        """Robust Guinier fit, with auto_gpa as fallback."""
        data = numpy.vstack((curve.q, curve.I, curve.sigma)).T
        try:
            return auto_guinier(data)
        except Exception:
            return auto_gpa(data)

    def est_shannon(curve, gui):
        res = dmax_shannon(curve.q, curve.I, gui.Rg, gui.I0)
        return res.Dmax, {"converged": res.converged, "nsol": res.nsolution,
                          "sigma": res.sigma_Dmax, "nmax": res.nmax}

    def est_shannon_exact(curve, gui):
        res = dmax_shannon(curve.q, curve.I, curve.Rg, curve.I0)
        return res.Dmax, {"converged": res.converged, "nsol": res.nsolution,
                          "sigma": res.sigma_Dmax, "nmax": res.nmax}

    def est_dnn(curve, gui):
        rg_nn, dmax_nn = dnn(curve.q, curve.I)
        return float(dmax_nn), {"rg_nn": float(rg_nn)}

    def est_three_rg(curve, gui):
        return 3.0 * gui.Rg, {}

    estimators = {
        "shannon": est_shannon,
        "shannon_ex": est_shannon_exact,
        "dnn": est_dnn,
        "three_rg": est_three_rg,
    }

    if use_bift:
        from freesas.bift import auto_bift

        def est_bift(curve, gui):
            data = numpy.vstack((curve.q, curve.I, curve.sigma)).T
            bo = auto_bift(data, npt=64, scan_size=11)
            key, value, nvalid = bo.get_best()
            return key.Dmax, {"nvalid": nvalid, "nevidence": len(bo.evidence_cache)}

        estimators["bift"] = est_bift
    return estimators, guinier


def run(use_bift=False, bift_every=15, out="results.csv"):
    estimators, guinier = build_estimators(use_bift)
    rows = []
    case_id = 0
    for shape, kwargs in SHAPES:
        for Dmax in DMAX_VALUES:
            for noise in NOISE_LEVELS:
                for seed in SEEDS:
                    if noise == 0.0 and seed != SEEDS[0]:
                        continue  # noiseless curves are deterministic
                    case_id += 1
                    curve = make_curve(
                        shape=shape, Dmax=Dmax, qmin=QMIN, qmax=QMAX, npt=NPT,
                        rel_error_at_qmax=noise, seed=seed, **kwargs,
                    )
                    try:
                        gui = guinier(curve)
                    except Exception as err:
                        logger.warning("Guinier failed on %s: %s", curve.label, err)
                        continue
                    base = {
                        "shape": shape,
                        "params": ";".join(f"{k}={v:g}" for k, v in kwargs.items()),
                        "Dmax_true": Dmax,
                        "Rg_true": curve.Rg,
                        "dmax_over_rg": Dmax / curve.Rg,
                        "noise": noise,
                        "seed": seed,
                        "Rg_guinier": gui.Rg,
                        "rg_gui_err": (gui.Rg - curve.Rg) / curve.Rg,
                        "nchannel_max": int(QMAX * Dmax / numpy.pi),
                        # is the true answer even inside the paper's scan window?
                        "in_window": 2.5 * gui.Rg <= Dmax <= 3.5 * gui.Rg,
                    }
                    for name, func in estimators.items():
                        if name == "bift" and case_id % bift_every:
                            continue
                        t0 = time.perf_counter()
                        try:
                            value, extra = func(curve, gui)
                        except Exception as err:
                            logger.warning("%s failed on %s: %s", name, curve.label, err)
                            value, extra = numpy.nan, {"error": repr(err)}
                        dt = time.perf_counter() - t0
                        row = dict(base)
                        row.update({
                            "method": name,
                            "Dmax_est": value,
                            "rel_err": (value - Dmax) / Dmax,
                            "time_ms": 1000.0 * dt,
                        })
                        row.update({f"x_{k}": v for k, v in extra.items()})
                        rows.append(row)
                    if case_id % 25 == 0:
                        print(f"  ... {case_id} cases", file=sys.stderr)

    fields = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows written to {out}")
    return rows


def summarize(rows):
    """Print median and 90th percentile of |relative error| per method."""
    methods = sorted({r["method"] for r in rows})

    def stats(subset):
        out = {}
        for m in methods:
            errs = [abs(r["rel_err"]) for r in subset
                    if r["method"] == m and numpy.isfinite(r["rel_err"])]
            if errs:
                out[m] = (numpy.median(errs), numpy.percentile(errs, 90), len(errs))
        return out

    def show(title, subset):
        st = stats(subset)
        if not st:
            return
        print(f"\n{title}  (n={len(subset)//max(1,len(st))} cases)")
        print(f"  {'method':12s} {'median|err|':>12s} {'p90|err|':>10s} {'n':>5s}")
        for m, (med, p90, n) in sorted(st.items(), key=lambda kv: kv[1][0]):
            print(f"  {m:12s} {100*med:11.2f}% {100*p90:9.2f}% {n:5d}")

    show("ALL CASES", rows)
    for noise in NOISE_LEVELS:
        show(f"noise = {noise:g}", [r for r in rows if r["noise"] == noise])
    for shape in sorted({r["shape"] + " " + r["params"] for r in rows}):
        show(f"shape = {shape}", [r for r in rows
                                 if r["shape"] + " " + r["params"] == shape])
    show("Dmax <= 10 nm", [r for r in rows if r["Dmax_true"] <= 10])
    show("Dmax > 10 nm", [r for r in rows if r["Dmax_true"] > 10])
    show("true Dmax inside [2.5,3.5].Rg window",
         [r for r in rows if r["in_window"]])
    show("true Dmax OUTSIDE the window", [r for r in rows if not r["in_window"]])

    # convergence rate of equation 12
    sh = [r for r in rows if r["method"] == "shannon"]
    conv = [r for r in sh if r.get("x_converged")]
    if sh:
        print(f"\nequation 12 satisfied: {len(conv)}/{len(sh)} "
              f"({100.0*len(conv)/len(sh):.0f}%)")
        for tag, subset in (("converged", conv),
                           ("not converged", [r for r in sh if not r.get("x_converged")])):
            errs = [abs(r["rel_err"]) for r in subset if numpy.isfinite(r["rel_err"])]
            if errs:
                print(f"  {tag:15s} median|err| = {100*numpy.median(errs):.2f}% "
                      f"(n={len(errs)})")

    # timing
    print("\nmedian time per call")
    for m in methods:
        ts = [r["time_ms"] for r in rows if r["method"] == m]
        if ts:
            print(f"  {m:12s} {numpy.median(ts):8.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bift", action="store_true", help="also run auto_bift")
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()
    summarize(run(use_bift=args.bift, out=args.out))
