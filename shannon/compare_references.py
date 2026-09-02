#!/usr/bin/env python3
# coding: utf-8
"""Score the fast Dmax estimators against TWO independent references.

Merges:
  experimental.csv  fast estimators + the free_bift reference (sandbox/2025/*.out)
  atsas_gnom.csv    the GNOM reference (sandbox/2025/atsas/*.out, via datgnom)

The point of having two references: if BIFT and GNOM disagree with each other
by X %, then no fast estimator can be meaningfully scored below X %. That
disagreement is the noise floor of the whole exercise, and it has to be
measured before any estimator is praised or condemned.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

import csv
import sys

import numpy

METHODS = ["dnn", "shannon", "three_rg", "knee"]
LABEL = {"dnn": "neural network", "shannon": "Shannon published",
         "three_rg": "3.Rg (control)", "knee": "plateau onset (mine)"}
QMIN = 0.0851627639266574
LIMIT = numpy.pi / QMIN     # first Shannon channel must be measured


def load(path, keys):
    out = {}
    with open(path) as fd:
        for raw in csv.DictReader(fd):
            rec = {}
            for key in keys:
                value = raw.get(key, "")
                try:
                    rec[key] = float(value) if value else numpy.nan
                except ValueError:
                    rec[key] = numpy.nan
            rec["status"] = raw.get("status", "")
            out[raw["file"]] = rec
    return out


def merge():
    fast = load("experimental.csv",
                ["Dmax_bift", "Dmax_bift_err", "Rg_bift", "chi2", "qmin_Rg",
                 "Rg_gui", "Rg_dnn", "dnn", "shannon", "three_rg", "knee",
                 "dmax_over_rg"])
    gnom = load("atsas_gnom.csv",
                ["Dmax", "Rg_real", "Rg_autorg", "estimate", "I0_real"])
    rows = []
    for name, rec in fast.items():
        merged = dict(rec)
        merged["file"] = name
        g = gnom.get(name)
        if g and g.get("status") == "ok" and numpy.isfinite(g.get("Dmax", numpy.nan)):
            merged["Dmax_gnom"] = g["Dmax"]
            merged["Rg_gnom"] = g["Rg_real"]
            merged["Rg_autorg"] = g["Rg_autorg"]
            merged["gnom_estimate"] = g["estimate"]
        else:
            merged["Dmax_gnom"] = numpy.nan
        rows.append(merged)
    return rows


def deviations(rows, method, reference):
    dev = []
    for r in rows:
        est, ref = r.get(method, numpy.nan), r.get(reference, numpy.nan)
        if numpy.isfinite(est) and numpy.isfinite(ref) and ref > 0:
            dev.append((est - ref) / ref)
    return numpy.array([d for d in dev if numpy.isfinite(d)])


def table(title, rows, reference):
    if not rows:
        return
    print(f"\n{title}   (n = {len(rows)}, reference = {reference})")
    print(f"  {'estimator':22s} {'median|dev|':>11s} {'p90':>8s} {'bias':>8s} "
          f"{'<10%':>7s} {'<20%':>7s}")
    collected = []
    for method in METHODS:
        dev = deviations(rows, method, reference)
        if dev.size:
            collected.append((numpy.median(numpy.abs(dev)), method, dev))
    for _, method, dev in sorted(collected):
        absdev = numpy.abs(dev)
        print(f"  {LABEL[method]:22s} {100*numpy.median(absdev):10.2f}% "
              f"{100*numpy.percentile(absdev, 90):7.1f}% "
              f"{100*numpy.median(dev):+7.1f}% "
              f"{100*numpy.mean(absdev < 0.10):6.1f}% "
              f"{100*numpy.mean(absdev < 0.20):6.1f}%")


def main():
    rows = merge()
    have_gnom = [r for r in rows if numpy.isfinite(r["Dmax_gnom"])]
    print(f"{len(rows)} curves with a BIFT reference")
    print(f"{len(have_gnom)} of them also have a GNOM reference "
          f"({100*len(have_gnom)/len(rows):.0f}%)")

    # ---- the noise floor: how far apart are the two references? ----
    print("\n" + "=" * 72)
    print("HOW MUCH DO THE TWO REFERENCES AGREE WITH EACH OTHER?")
    print("=" * 72)
    both = [r for r in have_gnom if r["Dmax_bift"] > 0]
    dev = numpy.array([(r["Dmax_bift"] - r["Dmax_gnom"]) / r["Dmax_gnom"]
                       for r in both])
    dev = dev[numpy.isfinite(dev)]
    absdev = numpy.abs(dev)
    print(f"  BIFT vs GNOM on Dmax, all {dev.size} shared curves:")
    print(f"    median |deviation| {100*numpy.median(absdev):.2f}%   "
          f"p90 {100*numpy.percentile(absdev, 90):.1f}%   "
          f"median bias {100*numpy.median(dev):+.2f}% (BIFT larger)")
    print(f"    within 10 % of each other: {100*numpy.mean(absdev < 0.10):.1f}%   "
          f"within 20 %: {100*numpy.mean(absdev < 0.20):.1f}%")

    clean = [r for r in both if r["chi2"] < 2
             and r.get("gnom_estimate", 0) > 0.7
             and r["Dmax_bift"] < LIMIT]
    dev_clean = numpy.array([(r["Dmax_bift"] - r["Dmax_gnom"]) / r["Dmax_gnom"]
                             for r in clean])
    dev_clean = dev_clean[numpy.isfinite(dev_clean)]
    print(f"\n  restricted to curves both tools like "
          f"(BIFT chi2 < 2, GNOM estimate > 0.7, Dmax < {LIMIT:.0f} nm): "
          f"n = {dev_clean.size}")
    print(f"    median |deviation| {100*numpy.median(numpy.abs(dev_clean)):.2f}%   "
          f"p90 {100*numpy.percentile(numpy.abs(dev_clean), 90):.1f}%   "
          f"median bias {100*numpy.median(dev_clean):+.2f}%")
    print("\n  >>> no fast estimator can be scored below this floor.")

    # ---- estimators against each reference ----
    print("\n" + "=" * 72)
    print("ESTIMATORS AGAINST EACH REFERENCE")
    print("=" * 72)
    table("GNOM reference, GNOM happy (estimate > 0.7)",
          [r for r in have_gnom if r.get("gnom_estimate", 0) > 0.7], "Dmax_gnom")
    table("BIFT reference, BIFT happy (chi2 < 2)",
          [r for r in rows if r["chi2"] < 2], "Dmax_bift")

    print("\n" + "-" * 72)
    print("VALID DOMAIN: both references happy AND Dmax < pi/qmin")
    print(f"(first Shannon channel measured, i.e. Dmax < {LIMIT:.1f} nm)")
    print("-" * 72)
    table("against GNOM", clean, "Dmax_gnom")
    table("against BIFT", clean, "Dmax_bift")

    print("\n--- consensus subset: the two references agree within 10 % ---")
    consensus = [r for r in clean
                 if abs(r["Dmax_bift"] - r["Dmax_gnom"]) / r["Dmax_gnom"] < 0.10]
    print(f"  n = {len(consensus)} ({100*len(consensus)/max(1,len(clean)):.0f}% of the valid domain)")
    table("against GNOM, consensus only", consensus, "Dmax_gnom")

    # ---- Rg cross-check across four independent estimates ----
    print("\n" + "=" * 72)
    print("Rg FROM FOUR INDEPENDENT ROUTES (consensus subset)")
    print("=" * 72)
    print(f"  {'pair':34s} {'median dev':>11s} {'median|dev|':>12s} {'p90|dev|':>9s}")
    for name, a, b in (
        ("freesas auto_guinier vs ATSAS autorg", "Rg_gui", "Rg_autorg"),
        ("neural network vs ATSAS autorg", "Rg_dnn", "Rg_autorg"),
        ("BIFT P(r) vs GNOM P(r)", "Rg_bift", "Rg_gnom"),
        ("neural network vs GNOM P(r)", "Rg_dnn", "Rg_gnom"),
    ):
        dev = numpy.array([(r[a] - r[b]) / r[b] for r in consensus
                           if numpy.isfinite(r.get(a, numpy.nan))
                           and numpy.isfinite(r.get(b, numpy.nan)) and r[b] > 0])
        dev = dev[numpy.isfinite(dev)]
        if dev.size:
            print(f"  {name:34s} {100*numpy.median(dev):+10.2f}% "
                  f"{100*numpy.median(numpy.abs(dev)):11.2f}% "
                  f"{100*numpy.percentile(numpy.abs(dev), 90):8.1f}%")

    print("\n--- does each estimator beat the control? (consensus, vs GNOM) ---")
    for method in ("dnn", "shannon", "knee"):
        wins = [abs(r[method] - r["Dmax_gnom"]) < abs(r["three_rg"] - r["Dmax_gnom"])
                for r in consensus
                if numpy.isfinite(r.get(method, numpy.nan))
                and numpy.isfinite(r.get("three_rg", numpy.nan))]
        if wins:
            print(f"  {LABEL[method]:22s} {100*numpy.mean(wins):5.1f}% of curves "
                  f"(n={len(wins)})")


if __name__ == "__main__":
    sys.exit(main())
