#!/usr/bin/env python3
# coding: utf-8
"""Slice experimental.csv: agreement of each estimator with the BIFT reference."""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

import csv
import sys

import numpy

METHODS = ["knee", "dnn", "three_rg", "shannon"]
LABEL = {"knee": "plateau onset", "dnn": "neural net",
         "three_rg": "3.Rg (control)", "shannon": "Shannon published"}


def load(path="experimental.csv"):
    rows = []
    with open(path) as fd:
        for raw in csv.DictReader(fd):
            rec = {"file": raw["file"]}
            for key, value in raw.items():
                if key == "file":
                    continue
                try:
                    rec[key] = float(value) if value else numpy.nan
                except ValueError:
                    rec[key] = numpy.nan
            rows.append(rec)
    return rows


def stats(rows, method):
    """Relative deviation from the BIFT reference."""
    dev = [(r[method] - r["Dmax_bift"]) / r["Dmax_bift"]
           for r in rows
           if numpy.isfinite(r.get(method, numpy.nan)) and r["Dmax_bift"] > 0]
    dev = numpy.array([d for d in dev if numpy.isfinite(d)])
    if dev.size == 0:
        return None
    absdev = numpy.abs(dev)
    return {
        "n": dev.size,
        "median_abs": numpy.median(absdev),
        "p90_abs": numpy.percentile(absdev, 90),
        "bias": numpy.median(dev),
        "within10": numpy.mean(absdev < 0.10),
        "within20": numpy.mean(absdev < 0.20),
    }


def table(title, rows):
    if not rows:
        return
    print(f"\n{title}   (n = {len(rows)})")
    print(f"  {'estimator':18s} {'median|dev|':>11s} {'p90':>8s} {'bias':>8s} "
          f"{'<10%':>7s} {'<20%':>7s}")
    collected = []
    for method in METHODS:
        st = stats(rows, method)
        if st:
            collected.append((st["median_abs"], method, st))
    for _, method, st in sorted(collected):
        print(f"  {LABEL[method]:18s} {100*st['median_abs']:10.2f}% "
              f"{100*st['p90_abs']:7.1f}% {100*st['bias']:+7.1f}% "
              f"{100*st['within10']:6.1f}% {100*st['within20']:6.1f}%")


def main(path="experimental.csv"):
    rows = load(path)
    print(f"{len(rows)} curves parsed")

    good = [r for r in rows if r["chi2"] < 2.0]
    print(f"BIFT reference usable (chi2 < 2): {len(good)}/{len(rows)} "
          f"({100*len(good)/len(rows):.0f}%)")

    table("ALL CURVES (unfiltered reference)", rows)
    table("REFERENCE chi2 < 2", good)
    table("REFERENCE chi2 < 1.2", [r for r in rows if r["chi2"] < 1.2])

    print("\n--- stratified by object size (chi2 < 2) ---")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)):
        subset = [r for r in good if lo <= r["Dmax_bift"] < hi]
        tag = f"Dmax {lo}-{hi} nm" if hi < 1e9 else f"Dmax > {lo} nm"
        table(tag, subset)

    print("\n--- stratified by shape ratio Dmax/Rg (chi2 < 2) ---")
    print("    the published method scans only [2.5, 3.5] x Rg")
    for lo, hi in ((0, 2.5), (2.5, 3.5), (3.5, 1e9)):
        subset = [r for r in good if lo <= r["dmax_over_rg"] < hi]
        tag = f"Dmax/Rg in [{lo}, {hi})" if hi < 1e9 else f"Dmax/Rg > {lo}"
        table(tag, subset)
    inside = numpy.mean([2.5 <= r["dmax_over_rg"] < 3.5 for r in good])
    print(f"\n  fraction of curves inside the published window: {100*inside:.1f}%")

    print("\n--- stratified by availability of a Guinier region (chi2 < 2) ---")
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 1e9)):
        subset = [r for r in good if lo <= r["qmin_Rg"] < hi]
        tag = f"qmin.Rg in [{lo}, {hi})" if hi < 1e9 else f"qmin.Rg > {lo}"
        table(tag, subset)

    print("\n--- is the knee calibration still valid on experimental data? ---")
    ratios = numpy.array([r["knee"] / r["Dmax_bift"] for r in good
                          if numpy.isfinite(r.get("knee", numpy.nan))
                          and r["Dmax_bift"] > 0])
    ratios = ratios[numpy.isfinite(ratios)]
    print(f"  knee / Dmax_bift : median {numpy.median(ratios):.4f}  "
          f"p16 {numpy.percentile(ratios, 16):.4f}  "
          f"p84 {numpy.percentile(ratios, 84):.4f}   (n={ratios.size})")
    print(f"  calibration fitted on synthetics = 1.1495, i.e. an extra "
          f"x{1/numpy.median(ratios):.4f} would be needed here")
    rescaled = numpy.abs(ratios / numpy.median(ratios) - 1)
    print(f"  after re-centring: median|dev| {100*numpy.median(rescaled):.2f}%  "
          f"p90 {100*numpy.percentile(rescaled, 90):.1f}%")

    print("\n--- Rg cross-check: who agrees with whom (chi2 < 2) ---")
    for name, key in (("Guinier vs BIFT", "Rg_gui"), ("network vs BIFT", "Rg_dnn")):
        dev = numpy.array([(r[key] - r["Rg_bift"]) / r["Rg_bift"] for r in good
                           if numpy.isfinite(r.get(key, numpy.nan))
                           and r["Rg_bift"] > 0])
        dev = dev[numpy.isfinite(dev)]
        print(f"  {name:18s} median {100*numpy.median(dev):+6.2f}%  "
              f"median|dev| {100*numpy.median(numpy.abs(dev)):5.2f}%  "
              f"p90|dev| {100*numpy.percentile(numpy.abs(dev), 90):6.1f}%  (n={dev.size})")

    print("\n--- how often does each estimator beat the 3.Rg control? (chi2 < 2) ---")
    for method in ("knee", "dnn", "shannon"):
        wins = [abs(r[method] - r["Dmax_bift"]) < abs(r["three_rg"] - r["Dmax_bift"])
                for r in good
                if numpy.isfinite(r.get(method, numpy.nan))
                and numpy.isfinite(r.get("three_rg", numpy.nan))]
        if wins:
            print(f"  {LABEL[method]:18s} {100*numpy.mean(wins):5.1f}% of curves "
                  f"(n={len(wins)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experimental.csv")
