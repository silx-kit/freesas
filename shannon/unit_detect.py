#!/usr/bin/env python3
# coding: utf-8
"""Detect whether q is in 1/A or 1/nm, by cross-checking the network against Guinier.

The problem: SASBDB serves both conventions and the file rarely says which.
SASDFX7.dat does carry "REMARK 265  q : defined in inverse Angstroms", but
load_scattering_data() does not read remarks, and most files carry nothing.

Why no shape-based test can work: under q -> 10 q every dimensionless product
is invariant -- q.Rg, q.Dmax, the oversampling ratio, the number of Shannon
channels. So the unit cannot be recovered from the shape of the curve alone;
it needs an absolute length scale from somewhere.

Two sources of such a scale are available in FreeSAS:

  magnitude   a prior on real instruments: BioSAXS q_max sits around 0.15-0.8
              1/A, i.e. 1.5-8 1/nm. A threshold on q_max separates the two
              conventions for most datasets, and fails in the overlap.

  network     freesas.dnn is *hard-wired* to 1/nm: preprocess() interpolates
              onto a fixed 1024-point grid over [0, 4] 1/nm. Feed it the wrong
              unit and its Rg disagrees violently with the Guinier Rg, which is
              itself unit-covariant. That disagreement is the signal.

The second test turns the network's unit-sensitivity -- a liability when used
blind -- into a unit detector.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

from collections import namedtuple

import numpy

UNIT_RESULT = namedtuple("UNIT_RESULT", "unit factor ratio_nm ratio_angstrom margin")


def _UNIT_RESULT_repr(self):
    return (f"q in 1/{self.unit} (x{self.factor:g}) "
            f"ratio_nm={self.ratio_nm:.3f} ratio_A={self.ratio_angstrom:.3f} "
            f"margin={self.margin:.1f}x")


UNIT_RESULT.__repr__ = _UNIT_RESULT_repr


def detect_by_magnitude(q):
    """Guess the unit from the sheer magnitude of q_max. Cheap, no model."""
    return "A" if q[-1] < 1.2 else "nm"


def detect_by_network(q, I, Rg_raw, dnn):
    """Guess the unit by asking which interpretation the network agrees with.

    :param q: scattering vector, in whichever unit
    :param I: scattering intensity
    :param Rg_raw: Rg from the Guinier fit, in the inverse unit of q
    :param dnn: a KerasDNN instance, hard-wired to 1/nm
    :return: UNIT_RESULT

    Hypothesis "nm": q is already in 1/nm, so Rg_raw is in nm and the network
    should reproduce it from q as-is.
    Hypothesis "A": q is in 1/A, so the network must be fed 10.q and should
    reproduce Rg_raw/10.
    """
    rg_as_nm = float(dnn(q, I)[0])
    rg_as_angstrom = float(dnn(10.0 * q, I)[0])

    # agreement of each hypothesis with the unit-covariant Guinier Rg
    ratio_nm = rg_as_nm / Rg_raw if Rg_raw else numpy.inf
    ratio_angstrom = rg_as_angstrom / (Rg_raw / 10.0) if Rg_raw else numpy.inf

    def badness(ratio):
        if not numpy.isfinite(ratio) or ratio <= 0:
            return numpy.inf
        return max(ratio, 1.0 / ratio)  # symmetric in log

    bad_nm, bad_a = badness(ratio_nm), badness(ratio_angstrom)
    if bad_nm <= bad_a:
        unit, factor = "nm", 1.0
        margin = bad_a / bad_nm if bad_nm else numpy.inf
    else:
        unit, factor = "A", 10.0
        margin = bad_nm / bad_a if bad_a else numpy.inf
    return UNIT_RESULT(unit, factor, ratio_nm, ratio_angstrom, margin)


def main():
    from freesas.autorg import auto_guinier, auto_gpa
    from freesas.dnn import KerasDNN
    from freesas.resources import resource_filename
    from freesas.sasio import load_scattering_data
    from freesas.test.utilstest import get_datafile
    from synthetic import make_curve

    dnn = KerasDNN(resource_filename("keras_models/Rg+Dmax.keras"))

    def guinier(data):
        try:
            return auto_guinier(data)
        except Exception:
            return auto_gpa(data)

    print("=== real files ===")
    print(f"{'file':20s} {'q_max':>8s} {'Rg_raw':>8s} {'truth':>6s} "
          f"{'magnitude':>10s} {'network':>8s} {'margin':>8s}")
    truths = {"bsa_005_sub.dat": "nm", "SASDF52.dat": "nm",
              "SASDFX7.dat": "A", "SASDFN8.dat": "nm"}
    files = {n: get_datafile(n) for n in
             ("bsa_005_sub.dat", "SASDF52.dat", "SASDFX7.dat")}
    files["SASDFN8.dat"] = "/home/kieffer/.claude/jobs/0356cb97/tmp/SASDFN8.dat"
    for name, path in files.items():
        data = load_scattering_data(path)
        q, I, s = data.T
        g = guinier(data)
        mag = detect_by_magnitude(q)
        net = detect_by_network(q, I, g.Rg, dnn)
        ok_m = "ok" if mag == truths[name] else "WRONG"
        ok_n = "ok" if net.unit == truths[name] else "WRONG"
        print(f"{name:20s} {q[-1]:8.3f} {g.Rg:8.3f} {truths[name]:>6s} "
              f"{mag + ' ' + ok_m:>10s} {net.unit + ' ' + ok_n:>8s} {net.margin:7.1f}x")

    print("\n=== synthetic curves, both conventions ===")
    print("  (built in nm, then re-expressed in 1/A by dividing q by 10)")
    print(f"{'case':30s} {'convention':>10s} {'q_max':>8s} "
          f"{'magnitude':>10s} {'network':>10s} {'margin':>8s}")
    nmag = nnet = total = 0
    for shape, kw, Dmax in (("sphere", {}, 5.0), ("sphere", {}, 10.0),
                            ("sphere", {}, 20.0), ("sphere", {}, 40.0),
                            ("spheroid", {"epsilon": 2.0}, 10.0),
                            ("core_shell", {"epsilon": 1.5, "X": 0.16}, 12.0),
                            ("parabola", {}, 8.0)):
        base = make_curve(shape=shape, Dmax=Dmax, qmin=0.05, qmax=4.0, npt=1024,
                          rel_error_at_qmax=0.05, seed=0, **kw)
        for convention, factor in (("nm", 1.0), ("A", 0.1)):
            q = base.q * factor
            g = guinier(numpy.vstack((q, base.I, base.sigma)).T)
            mag = detect_by_magnitude(q)
            net = detect_by_network(q, base.I, g.Rg, dnn)
            total += 1
            nmag += mag == convention
            nnet += net.unit == convention
            print(f"{shape + ' ' + str(Dmax):30s} {convention:>10s} {q[-1]:8.3f} "
                  f"{mag + (' ok' if mag == convention else ' WRONG'):>10s} "
                  f"{net.unit + (' ok' if net.unit == convention else ' WRONG'):>10s} "
                  f"{net.margin:7.1f}x")
    print(f"\n  magnitude test: {nmag}/{total}   network test: {nnet}/{total}")


if __name__ == "__main__":
    main()
