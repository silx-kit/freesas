#!/usr/bin/env python3
# coding: utf-8
"""Sensitivity of the Dmax estimators to the accessible q-range.

The main benchmark samples q up to 4 nm-1, which is exactly the grid the
neural network was trained on. Real BioSAXS datasets often stop much earlier
(SASDFX7.dat in the FreeSAS test set stops at 0.37 nm-1), so this axis
deserves its own experiment: it is the one place where the two approaches
differ structurally rather than by a few percent.

The Shannon approach has an intrinsic requirement -- it needs at least a few
channels n.pi/Dmax inside the measured range -- whereas the network has an
extrinsic one: its input grid is hard-coded to 1024 points over [0, 4] nm-1
by freesas.dnn.preprocess(), and anything outside is padded with zeros.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

import numpy

from synthetic import make_curve
from nsdmax import dmax_shannon


def main():
    from freesas.autorg import auto_guinier, auto_gpa
    from freesas.dnn import KerasDNN
    from freesas.resources import resource_filename

    dnn = KerasDNN(resource_filename("keras_models/Rg+Dmax.keras"))

    print("Dmax = 10 nm sphere, noise = 5%, 1024 points, varying qmax")
    print(f"{'qmax':>6s} {'q.Dmax/pi':>10s} {'3Rg':>8s} {'shannon':>9s} "
          f"{'sh.conv':>8s} {'dnn':>8s} {'dnn_Rg':>8s}")
    for qmax in (0.4, 0.8, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        curve = make_curve(shape="sphere", Dmax=10.0, qmin=0.05, qmax=qmax,
                           npt=1024, rel_error_at_qmax=0.05, seed=0)
        data = numpy.vstack((curve.q, curve.I, curve.sigma)).T
        try:
            gui = auto_guinier(data)
        except Exception:
            gui = auto_gpa(data)
        res = dmax_shannon(curve.q, curve.I, gui.Rg, gui.I0)
        rg_nn, dmax_nn = dnn(curve.q, curve.I)
        nchan = qmax * curve.Dmax / numpy.pi
        print(f"{qmax:6.1f} {nchan:10.1f} {3*gui.Rg:8.2f} {res.Dmax:9.2f} "
              f"{str(res.converged):>8s} {dmax_nn:8.2f} {rg_nn:8.2f}")

    print("\nSame, but Dmax swept at the network's native qmax = 4 nm-1")
    print(f"{'Dmax':>6s} {'nchannel':>9s} {'Rg_true':>8s} {'Rg_gui':>8s} "
          f"{'Rg_dnn':>8s} {'3Rg':>8s} {'shannon':>9s} {'dnn':>8s}")
    for Dmax in (2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0):
        curve = make_curve(shape="sphere", Dmax=Dmax, qmin=min(0.05, 0.3 / Dmax),
                           qmax=4.0, npt=1024, rel_error_at_qmax=0.05, seed=0)
        data = numpy.vstack((curve.q, curve.I, curve.sigma)).T
        try:
            gui = auto_guinier(data)
        except Exception:
            try:
                gui = auto_gpa(data)
            except Exception:
                print(f"{Dmax:6.1f}  Guinier fit failed")
                continue
        res = dmax_shannon(curve.q, curve.I, gui.Rg, gui.I0)
        rg_nn, dmax_nn = dnn(curve.q, curve.I)
        print(f"{Dmax:6.1f} {4.0*Dmax/numpy.pi:9.1f} {curve.Rg:8.2f} "
              f"{gui.Rg:8.2f} {rg_nn:8.2f} {3*gui.Rg:8.2f} {res.Dmax:9.2f} "
              f"{dmax_nn:8.2f}")


if __name__ == "__main__":
    main()
