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
                 Guinier=None, ift=None, title="Scattering curve",
                 ax=None, labelsize=None, fontsize=None):
    """
    Generate a scattering plot I = f(q) in semi_log_y.
    
    :param data: data read from an ASCII file, 3 column (q,I,err)
    :param filename: name of the file where the cuve should be saved
    :param format: image format
    :param unit: Unit name for Rg and 1/q
    :param Guinier: output of autoRg
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
    assert data.shape[1] >= 2
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

    if (Guinier is None):
        if (ift is not None):
            # best = ift.calc_stats()[0]
            I0 = Guinier.I0
            rg = Guinier.rg
            first_point = ift.high_start
            last_point = ift.high.stop
        else:
            rg = I0 = first_point = last_point = None
    else:
        I0 = Guinier.I0
        rg = Guinier.Rg
        first_point = Guinier.start_point
        last_point = Guinier.end_point

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


def Kratky_plot(data, filename=None, format="svg", unit="nm",
                 Guinier, title="Dimensionless Kratky plot - $R_{g}$ ",
                 ax=None, labelsize=None, fontsize=None):
    """
    Generate a Kratky plot q²Rg²I/I₀ = f(q·Rg)
    
    :param data: data read from an ASCII file, 3 column (q,I,err)
    :param filename: name of the file where the cuve should be saved
    :param format: image format
    :param unit: Unit name for Rg and 1/q
    :param Guinier: output of autoRg
    :param ax: subplot where the plot shall go in
    :return: the matplotlib figure
    """
    label = "Experimental data"
    assert data.ndim == 2
    assert data.shape[1] >= 2
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
    Rg = Guinier.Rg
    I0 = Guinier.I0

    xdata = q * Rg
    ydata = xdata * xdata * I / I0
    if err is not None:
        dy = xdata * xdata * err / I0
        dplot = ax.errorbar(xdata, ydata, dy, label=label, capsize=0, color="blue", ecolor="lightblue")
    else:
        dplot = ax.plot(xdata, ydata, label=label, color="blue")
    ax.set_ylabel('$(qR_{g})^2 I/I_{0}$', fontsize=fontsize)
    ax.set_xlabel('$qR_{g}$', fontsize=fontsize)
    ax.legend(loc=1)

    ax.hlines(3.0 * numpy.exp(-1), xmin=-0.05, xmax=max(xdata), color='0.75', linewidth=1.0)
    ax.vlines(numpy.sqrt(3.0), ymin=-0.01, ymax=max(ydata), color='0.75', linewidth=1.0)
    ax.set_xlim(xmin=-0.05, xmax=8.5)
    ax.set_ylim(ymin=-0.01, ymax=3.5)
    ax.set_title(title)
#     ax.legend([dplot[0]], [dplot[0].get_label()], loc=0)
    ax.legend(loc=0)
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    if filename:
        if format:
            fig.savefig(filename, format=format)
        else:
            fig.savefig(filename)
    return fig


def guinier_plot(data, Guinier, filename=None,
                 format="png", unit="nm", ax=None, labelsize=None, fontsize=None):
    """
    Generate a Guinier plot: ln(I) = f(q²)
    
    :param data: data read from an ASCII file, 3 column (q,I,err)
    :param Guinier: A RG_RESULT object from AutoRg
    :param  filename: name of the file where the cuve should be saved
    :param format: image format
    :param: ax: subplot where to plot in
    :return: the matplotlib figure
    """
    assert data.ndim == 2
    assert data.shape[1] >= 2

    q, I, err = data.T[:3]

    mask = (I > 0) & numpy.isfinite(I) & (q > 0) & numpy.isfinite(q)
    if err is not None:
        mask &= (err > 0.0) & numpy.isfinite(err)
    mask = mask.astype(bool)
    Rg = Guinier.Rg
    I0 = Guinier.I0
    first_point = Guinier.start_point
    last_point = Guinier.end_point
    intercept = numpy.log(I0)
    slope = -Rg * Rg / 3.0
    end = numpy.where(q > 1.5 / Rg)[0][0]

    q2 = q[:end] ** 2
    logI = numpy.log(I[:end])

    if ax:
        fig = ax.figure
    else:
        fig, ax = subplots(figsize=(12, 10))
    if err is not None:
        dlogI = err[:end] / logI
        ax.errorbar(q2[mask], logI[mask], dlogI[mask], label="Experimental curve",
                    capsize=0, color="blue", ecolor="lightblue")
    else:
        ax.plot(q2[mask], logI[mask], label="Experimental curve", color="blue")
    # ax.plot(q2[first_point:last_point], logI[first_point:last_point], marker='D', markersize=5, label="Guinier region")
    xmin = q2[first_point]
    xmax = q2[last_point]
    ymax = logI[first_point]
    ymin = logI[last_point]
    dy = (ymax - ymin) / 2.0
    ax.vlines(xmin, ymin=ymin, ymax=ymax + dy, color='0.75', linewidth=1.0)
    ax.vlines(xmax, ymin=ymin - dy, ymax=ymin + dy, color='0.75', linewidth=1.0)
    ax.annotate("$(qR_{g})_{min}$=%.1f" % (Rg * q[first_point]), (xmin, ymax + dy),
                xytext=None, xycoords='data', textcoords='data')
    ax.annotate("$(qR_{g})_{max}$=%.1f" % (Rg * q[last_point]), (xmax, ymin + dy),
                xytext=None, xycoords='data', textcoords='data')
    ax.annotate("Guinier region", (xmin, ymin - dy),
                xytext=None, xycoords='data', textcoords='data')
    ax.plot(q2, intercept + slope * q2, label="ln[$I(q)$] = %.2f %.2f * $q^2$" % (intercept, slope), color="red")
    ax.set_ylabel('ln[$I(q)$]', fontsize=fontsize)
    ax.set_xlabel('$q^2$ (%s$^{-2}$)' % unit, fontsize=fontsize)
    ax.set_title("Guinier plot: $R_{g}=$%.1f %s $I_{0}=$%.1f" % (Rg, unit, I0))
    ax.legend()
    if filename:
        if format:
            fig.savefig(filename, format=format)
        else:
            fig.savefig(filename)
    return fig


def density_plot(ift, filename=None, format="png", unit="nm",
                 ax=None, labelsize=None, fontsize=None):
    """
    Generate a density plot p(r)
     
    @param ift: An IFT result comming out of BIFT
    @param  filename: name of the file where the cuve should be saved
    @param format: image format
    @param ax: subplotib where to plot in
    @return: the matplotlib figure
    """
    if ax:
        fig = ax.figure
    else:
        fig, ax = subplots(figsize=(12, 10))

    ax.errorbar(ift.radius, ift.density, out["P(r)_err"], label="Density", capsize=0, color="blue", ecolor="lightblue")
    ax.set_ylabel('$\\rho (r)$')
    ax.set_xlabel('$r$ (%s)' % unit)
    ax.set_title("Pair distribution function")
    ax.legend()
    if filename:
        if format:
            fig.savefig(filename, format=format)
        else:
            fig.savefig(filename)
    return fig


densityPlot = density_plot
