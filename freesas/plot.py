# -*- coding: utf-8 -*-
"""
Functions to generating graphs related to 
"""

__authors__ = ["Jerome Kieffer"]
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "12/05/2020"

import logging
logger = logging.getLogger(__name__)
import numpy
from matplotlib.pyplot import subplots


def scatter_plot(data, filename=None, format="svg", unit="nm",
                 guinier=None, ift=None, title="Scattering curve",
                 ax=None, labelsize=None, fontsize=None):
    """
    Generate a scattering plot I = f(q) in semi_log_y.
    
    :param data: data read from an ASCII file, 3 column (q,I,err)
    :param filename: name of the file where the cuve should be saved
    :param format: image format
    :param unit: Unit name for Rg and 1/q
    :param guinier: output of autoRg
    :param ift: converged instance of BIFT (output of auto_bift)
    :param ax: subplot where the plot shall go in
    :return: the matplotlib figure
    """
    label_exp = "Experimental data"
    label_guinier = "Guinier region"
    label_ift = "BIFT extraplolated"
    exp_color = "blue"
    err_color = "lightblue"
    guinier_color = "limegreen"
    ift_color = "crimson"
    assert data.ndim == 2
    assert data.shape[1] in [2, 3]
    q = data.T[0]
    I = data.T[1]
    try:
        err = data.T[2]
    except:
        err = None
    if ax:
        fig = ax.figure
    else:
        fig, ax = subplots()

    # Extend q to zero
    delta_q = (q[-1] - q[0]) / (len(q) - 1)
    extra_q = int(q[0] / delta_q)
    first = q[0] - extra_q * delta_q
    q_ext = numpy.linspace(first, q[-1], extra_q + len(q))

    if (guinier is None):
        if (ift is not None):
            # best = ift.calc_stats()[0]
            I0 = guinier.I0
            rg = guinier.rg
            first_point = ift.high_start
            last_point = ift.high.stop
        else:
            rg = I0 = first_point = last_point = None
    else:
        I0 = guinier.I0
        rg = guinier.Rg
        first_point = guinier.start_point
        last_point = guinier.end_point

    if (rg is None) and (ift is None):
        if err is not None:
            ax.errorbar(q, I, err, label=label_exp, capsize=0, color=exp_color, ecolor=err_color)
        else:
            ax.plot(q, I, label=label_exp, color="blue")
    else:
        q_guinier = q[first_point:last_point]
        I_guinier = I0 * numpy.exp(-(q_guinier * rg) ** 2 / 3)
        if err is not None:
            ax.errorbar(q, I, err, label=label_exp, capsize=0, color=exp_color, ecolor=err_color, alpha=0.5)
        else:
            ax.plot(q, I, label=label_exp, color=exp_color, alpha=0.5)
        label_guinier += ": $R_g=$%.2f%s, $I_0=$%.2f" % (rg, unit, I0)
        ax.plot(q_guinier, I_guinier, label=label_guinier, color=guinier_color, linewidth=5)

    if ift:
        stats = ift.calc_stats()
        r = stats.radius
        T = numpy.outer(q_ext, r / numpy.pi)
        T = (4 * numpy.pi * (r[-1] - r[0]) / (len(r) - 1)) * numpy.sinc(T)
        p = stats.density_avg
        label_ift += ": $D_{max}=$%.2f %s, $R_g=$%.2f %s, $I_0=$%.2f" % (stats.Dmax_avg, unit, stats.Rg_avg, unit, stats.I0_avg)
        ax.plot(q_ext, T.dot(p), label=label_ift, color=ift_color)

    ax.set_ylabel('$I(q)$ (log scale)', fontsize=fontsize)
    ax.set_xlabel('$q$ (%s$^{-1}$)' % unit, fontsize=fontsize)
    ax.set_title(title)
    ax.set_yscale("log")
#     ax.set_ylim(ymin=I.min() * 10, ymax=I.max() * 1.1)

    # Re-order labels ...
    crv, lab = ax.get_legend_handles_labels()
    ordered_lab = []
    ordered_crv = []
    for l in [label_exp, label_guinier, label_ift]:
        idx = lab.index(l)
        ordered_lab.append(lab[idx])
        ordered_crv.append(crv[idx])
    ax.legend(ordered_crv, ordered_lab, loc=3)
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    if filename:
        if format:
            fig.savefig(filename, format=format)
        else:
            fig.savefig(filename)
    return fig
