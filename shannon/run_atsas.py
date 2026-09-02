#!/usr/bin/env python3
# coding: utf-8
"""Generate GNOM P(r) profiles for sandbox/2025/atsas/*.dat with the ATSAS suite.

Pipeline, per curve:
    autorg  -f csv <file>              -> Rg (in nm, since q is in 1/nm)
    datgnom -r <Rg> -o <file>.out      -> GNOM .out with P(r) and Dmax

This provides a reference for Dmax that is *independent of BIFT*, which matters:
on the very first test curve GNOM gives Dmax = 16.15 nm where free_bift gives
20.47 nm, a 21 % disagreement. Comparing the fast estimators against two
independent references separates "the estimator is wrong" from "the references
themselves disagree".

Writes the .out files in place and a summary CSV of the parsed headers.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

import csv
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

ATSAS = "/home/kieffer/ATSAS-4.1.4-1"
DATA_DIR = "/home/kieffer/workspace/freesas/sandbox/2025/atsas"
TIMEOUT = 120

ENV = dict(os.environ)
ENV["ATSAS"] = ATSAS
ENV["PATH"] = os.path.join(ATSAS, "bin") + os.pathsep + ENV.get("PATH", "")

_RE = {
    "Dmax": re.compile(r"^\s*Maximum characteristic size:\s+([-\d.eE+]+)"),
    "Rg_real": re.compile(r"^\s*Real space Rg:\s+([-\d.eE+]+)\s*\+-\s*([-\d.eE+]+)"),
    "I0_real": re.compile(r"^\s*Real space I\(0\):\s+([-\d.eE+]+)\s*\+-\s*([-\d.eE+]+)"),
    "Rg_recip": re.compile(r"^\s*Reciprocal space Rg:\s+([-\d.eE+]+)"),
    "I0_recip": re.compile(r"^\s*Reciprocal space I\(0\):\s+([-\d.eE+]+)"),
    "estimate": re.compile(r"^\s*Total Estimate:\s+([-\d.eE+]+)"),
}


def run_autorg(path):
    """Return Rg from ATSAS autorg, or None when no Guinier region is found."""
    try:
        proc = subprocess.run(
            [os.path.join(ATSAS, "bin", "autorg"), "-f", "csv", path],
            capture_output=True, text=True, env=ENV, timeout=TIMEOUT)
    except subprocess.SubprocessError:
        return None
    for line in proc.stdout.splitlines():
        fields = line.split(",")
        # skip the header line, which starts with "File"
        if len(fields) < 4 or fields[0].strip() in ("File", ""):
            continue
        try:
            rg = float(fields[1])
        except ValueError:
            continue
        return rg if rg > 0 else None
    return None


def run_datgnom(path, rg, out_path):
    """Run datgnom; return True when an .out was produced."""
    try:
        subprocess.run(
            [os.path.join(ATSAS, "bin", "datgnom"), "-r", f"{rg:.6g}",
             "-o", out_path, path],
            capture_output=True, text=True, env=ENV, timeout=TIMEOUT)
    except subprocess.SubprocessError:
        return False
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def parse_gnom(out_path):
    """Pull Dmax and the moments out of a GNOM .out header."""
    rec = {}
    try:
        with open(out_path, errors="replace") as fd:
            for line in fd:
                if line.lstrip().startswith("####      Real Space Data"):
                    break
                for key, pattern in _RE.items():
                    match = pattern.match(line)
                    if match:
                        rec[key] = float(match.group(1))
                        if match.lastindex and match.lastindex > 1:
                            rec[key + "_err"] = float(match.group(2))
    except OSError:
        return rec
    return rec


def process(path):
    """autorg then datgnom for one curve."""
    out_path = path[:-4] + ".out"
    row = {"file": os.path.basename(path)}
    rg = run_autorg(path)
    if rg is None:
        row["status"] = "autorg_failed"
        return row
    row["Rg_autorg"] = rg
    if not run_datgnom(path, rg, out_path):
        row["status"] = "datgnom_failed"
        return row
    rec = parse_gnom(out_path)
    if "Dmax" not in rec:
        row["status"] = "unparsable"
        return row
    row.update(rec)
    row["status"] = "ok"
    return row


def main(limit=None, workers=4, out_csv="atsas_gnom.csv"):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.dat")))
    if limit:
        paths = paths[:limit]
    print(f"{len(paths)} curves, {workers} workers", file=sys.stderr)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(process, paths, chunksize=16)):
            rows.append(row)
            if (index + 1) % 250 == 0:
                ok = sum(r.get("status") == "ok" for r in rows)
                print(f"  {index + 1}/{len(paths)}  ok={ok}", file=sys.stderr)

    fields = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    tally = Counter(r.get("status") for r in rows)
    print(f"\n{len(rows)} processed -> {out_csv}", file=sys.stderr)
    for status, count in tally.most_common():
        print(f"  {status}: {count} ({100 * count / len(rows):.1f}%)", file=sys.stderr)
    return rows


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
