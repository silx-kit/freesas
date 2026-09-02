#!/usr/bin/env python3
# coding: utf-8
"""Validation of the Dmax estimators on 5416 experimental BM29 curves.

Data: sandbox/2025/*.dat (q in 1/nm, 1000 points, q = 0.085 to 4.96) with the
matching *.out produced by free_bift, which carries Dmax, alpha, chi2, logP, Rg
and I0 plus the P(r) curve.

Two things to keep in mind when reading the numbers:

  * The reference Dmax comes from BIFT, so it is an *estimate*, not ground
    truth. What is measured here is agreement with the tool currently in
    production, not absolute accuracy. The synthetic benchmark remains the
    only place where truth is known.
  * These curves were recorded after the training set of the neural network,
    on the same instrument. For the network this is therefore a genuine
    held-out test, which the synthetic benchmark could not provide.

Quality filtering is essential: chi2 has a median of 1.03 but a p95 of 143.
A reference from a fit that did not converge says nothing about any estimator.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

import csv
import glob
import logging
import os
import re
import sys
import time

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsdmax import dmax_shannon          # noqa: E402
from knee import dmax_knee               # noqa: E402

logging.basicConfig(level=logging.ERROR)

KNEE_CALIBRATION = 1.1495   # frozen on the synthetic tuning set
KNEE_FACTOR = 1.1
DATA_DIR = "/home/kieffer/workspace/freesas/sandbox/2025"

_FIELDS = [("Dmax", "Dmax"), ("chi2", "χ²"), ("Rg", "Rg"),
           ("I0", "I₀"), ("logP", "logP"), ("alpha", "\U0001d6c2")]
_PATTERNS = {k: re.compile(r"^# " + re.escape(s) + r"=\s*([-\d.eE+]+)(?:±([-\d.eE+]+))?")
             for k, s in _FIELDS}


def parse_out(path):
    """Read the header of a free_bift .out file."""
    rec = {}
    with open(path) as fd:
        for line in fd:
            if not line.startswith("#"):
                break
            stripped = line.rstrip()
            for key, pattern in _PATTERNS.items():
                match = pattern.match(stripped)
                if match:
                    rec[key] = float(match.group(1))
                    err = match.group(2)
                    rec[key + "_err"] = float(err) if err else numpy.nan
    return rec


def collect():
    """Pair every .out with its .dat."""
    pairs = []
    for out in sorted(glob.glob(os.path.join(DATA_DIR, "*.out"))):
        dat = out[:-4] + ".dat"
        if os.path.exists(dat):
            pairs.append((dat, out))
    return pairs


def main(limit=None, out_csv="experimental.csv"):
    from freesas.autorg import auto_guinier, auto_gpa
    from freesas.dnn import KerasDNN
    from freesas.resources import resource_filename
    from freesas.sasio import load_scattering_data

    dnn = KerasDNN(resource_filename("keras_models/Rg+Dmax.keras"))
    pairs = collect()
    if limit:
        pairs = pairs[:limit]
    print(f"{len(pairs)} pairs found", file=sys.stderr)

    rows = []
    t_start = time.perf_counter()
    for index, (dat, out) in enumerate(pairs):
        ref = parse_out(out)
        if not ref.get("Dmax") or ref["Dmax"] <= 0 or not ref.get("Rg"):
            continue
        try:
            data = load_scattering_data(dat)
        except Exception:
            continue
        if data.ndim != 2 or data.shape[1] < 3 or data.shape[0] < 50:
            continue
        q, intensity, sigma = data.T[:3]

        row = {
            "file": os.path.basename(dat),
            "Dmax_bift": ref["Dmax"],
            "Dmax_bift_err": ref.get("Dmax_err", numpy.nan),
            "Rg_bift": ref["Rg"],
            "chi2": ref.get("chi2", numpy.nan),
            "logP": ref.get("logP", numpy.nan),
            "dmax_over_rg": ref["Dmax"] / ref["Rg"] if ref["Rg"] else numpy.nan,
            "qmin_Rg": q[0] * ref["Rg"],
        }

        # Guinier anchor, exactly as a user would obtain it
        try:
            gui = auto_guinier(data)
        except Exception:
            try:
                gui = auto_gpa(data)
            except Exception:
                gui = None
        if gui is None or gui.Rg <= 0:
            row["Rg_gui"] = numpy.nan
        else:
            row["Rg_gui"] = gui.Rg
            row["I0_gui"] = gui.I0
            row["three_rg"] = 3.0 * gui.Rg
            try:
                row["shannon"] = dmax_shannon(q, intensity, gui.Rg, gui.I0).Dmax
            except Exception:
                row["shannon"] = numpy.nan
            try:
                row["knee"] = KNEE_CALIBRATION * dmax_knee(
                    q, intensity, gui.Rg, gui.I0, factor=KNEE_FACTOR).Dmax
            except Exception:
                row["knee"] = numpy.nan
        # the network needs no anchor
        try:
            rg_nn, dmax_nn = dnn(q, intensity)
            row["dnn"] = float(dmax_nn)
            row["Rg_dnn"] = float(rg_nn)
        except Exception:
            row["dnn"] = numpy.nan

        rows.append(row)
        if (index + 1) % 250 == 0:
            rate = (time.perf_counter() - t_start) / (index + 1)
            print(f"  {index + 1}/{len(pairs)}  {1000 * rate:.0f} ms/curve",
                  file=sys.stderr)

    fields = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows written to {out_csv}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
