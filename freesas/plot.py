# -*- coding: utf-8 -*-
"""
Functions to generating graphs related to
"""

__authors__ = ["Jerome Kieffer"]
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "14/05/2020"

import logging
logger = logging.getLogger(__name__)
import numpy
from matplotlib.pyplot import subplots, switch_backend


def scatter_plot(data, guinier=None, ift=None,
                 filename=None, format="svg", unit="nm",
                 title="Scattering curve",
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
        label_guinier += ": $R_g=$%.2f %s, $I_0=$%.2f" % (rg, unit, I0)
        ax.plot(q_guinier, I_guinier, label=label_guinier, color=guinier_color, linewidth=5)

    if ift:
        from ._bift import BIFT, StatsResult
        if isinstance(ift, BIFT):
            stats = ift.calc_stats()
        elif isinstance(ift, StatsResult):
            stats = ift
        else:
            raise TypeError("ift is expected to be a BIFT object")

        r = stats.radius
        T = numpy.outer(q_ext, r / numpy.pi)
        T = (4 * numpy.pi * (r[-1] - r[0]) / (len(r) - 1)) * numpy.sinc(T)
        p = stats.density_avg
        label_ift += ": $D_{max}=$%.2f %s,\n    $R_g=$%.2f %s, $I_0=$%.2f" % (stats.Dmax_avg, unit, stats.Rg_avg, unit, stats.I0_avg)
        ax.plot(q_ext, T.dot(p), label=label_ift, color=ift_color)

    ax.set_ylabel('$I(q)$ (log scale)', fontsize=fontsize)
    ax.set_xlabel('$q$ (%s$^{-1}$)' % unit, fontsize=fontsize)
    ax.set_title(title)
    ax.set_yscale("log")
#     ax.set_ylim(ymin=I.min() * 10, top=I.max() * 1.1)

    # Re-order labels ...
    crv, lab = ax.get_legend_handles_labels()
    ordered_lab = []
    ordered_crv = []
    for l in [label_exp, label_guinier, label_ift]:
        try:
            idx = lab.index(l)
        except:
            continue 
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


def kratky_plot(data, guinier,
                filename=None, format="svg", unit="nm",
                title="Dimensionless Kratky plot",
                ax=None, labelsize=None, fontsize=None):
    """
    Generate a Kratky plot q²Rg²I/I₀ = f(q·Rg)

    :param data: data read from an ASCII file, 3 column (q,I,err)
    :param guinier: output of autoRg
    :param filename: name of the file where the cuve should be saved
    :param format: image format
    :param unit: Unit name for Rg and 1/q
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
    Rg = guinier.Rg
    I0 = guinier.I0

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
    ax.set_xlim(left=-0.05, right=8.5)
    ax.set_ylim(bottom=-0.01, top=(min(3.5, max(ydata))))
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


def guinier_plot(data, guinier, filename=None,
                 format="png", unit="nm", ax=None, labelsize=None, fontsize=None):
    """
    Generate a guinier plot: ln(I) = f(q²)

    :param data: data read from an ASCII file, 3 column (q,I,err)
    :param guinier: A RG_RESULT object from AutoRg
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
    Rg = guinier.Rg
    I0 = guinier.I0
    first_point = guinier.start_point
    last_point = guinier.end_point
    intercept = numpy.log(I0)
    slope = -Rg * Rg / 3.0
    end = numpy.where(q > 1.5 / Rg)[0][0]
    mask[end:] = False

    q2 = q[mask] ** 2
    logI = numpy.log(I[mask])

    if ax:
        fig = ax.figure
    else:
        fig, ax = subplots(figsize=(12, 10))
    if err is not None:
        dlogI = err[mask] / logI
        ax.errorbar(q2, logI, dlogI, label="Experimental curve",
                    capsize=0, color="blue", ecolor="lightblue",
                    alpha=0.5)
    else:
        ax.plot(q2[mask], logI[mask], label="Experimental curve", color="blue",
                alpha=0.5)
    # ax.plot(q2[first_point:last_point], logI[first_point:last_point], marker='D', markersize=5, label="guinier region")
    xmin = q[first_point] ** 2
    xmax = q[last_point] ** 2
    ymax = numpy.log(I[first_point])
    ymin = numpy.log(I[last_point])
    dy = (ymax - ymin) / 2.0
    ax.vlines(xmin, ymin=ymin, ymax=ymax + dy, color='0.75', linewidth=1.0)
    ax.vlines(xmax, ymin=ymin - dy, ymax=ymin + dy, color='0.75', linewidth=1.0)
    ax.annotate("$(qR_{g})_{min}$=%.1f" % (Rg * q[first_point]), (xmin, ymax + dy),
                xytext=None, xycoords='data', textcoords='data')
    ax.annotate("$(qR_{g})_{max}$=%.1f" % (Rg * q[last_point]), (xmax, ymin + dy),
                xytext=None, xycoords='data', textcoords='data')
    ax.annotate("Guinier region", (xmin, ymin - dy),
                xytext=None, xycoords='data', textcoords='data')
    ax.plot(q2[:end], intercept + slope * q2[:end], label="ln[$I(q)$] = %.2f %.2f * $q^2$" % (intercept, slope), color="crimson")
    ax.set_ylabel('ln[$I(q)$]', fontsize=fontsize)
    ax.set_xlabel('$q^2$ (%s$^{-2}$)' % unit, fontsize=fontsize)
    ax.set_title("Guinier plot: $R_{g}=$%.2f %s $I_{0}=$%.2f" % (Rg, unit, I0))
    ax.legend()
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

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

    from ._bift import BIFT, StatsResult
    if isinstance(ift, BIFT):
        stats = ift.calc_stats()
    elif isinstance(ift, StatsResult):
        stats = ift
    else:
        raise TypeError("ift is expected to be a BIFT object")

    ax.errorbar(ift.radius, ift.density_avg, ift.density_std,
                label="BIFT: χ$_{r}^{2}=$%.2f\n $D_{max}=$%.2f %s\n $R_{g}=$%.2f %s\n $I_{0}=$%.2f" % (stats.chi2r_avg, stats.Dmax_avg, unit, stats.Rg_avg, unit, stats.I0_avg),
                capsize=0, color="blue", ecolor="lightblue")
    ax.set_ylabel('$p(r)$', fontsize=fontsize)
    ax.set_xlabel('$r$ (%s)' % unit, fontsize=fontsize)
    ax.set_title("Pair distribution function")
    ax.legend()
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)

    if filename:
        if format:
            fig.savefig(filename, format=format)
        else:
            fig.savefig(filename)
    return fig


def plot_all(data, filename=None, format=None, unit="nm", labelsize=None, fontsize=None):
    from . import bift, autorg
    try:
        guinier = autorg.autoRg(data)
    except autorg.InsufficientDataError:
        raise
    logger.debug(guinier)
    try:
        bo = bift.auto_bift(data, npt=100, scan_size=11, Dmax_over_Rg=3)
    except (autorg.InsufficientDataError,
            autorg.NoGuinierRegionError,
            ValueError):
        raise
    else:
        ift = bo.calc_stats()
    logger.debug(ift)
    fig, ax = subplots(2, 2, figsize=(12, 10))
    scatter_plot(data, guinier=guinier, ift=ift, ax=ax[0, 0], unit=unit, labelsize=labelsize, fontsize=fontsize)
    guinier_plot(data, guinier, filename=None, format=None, unit=unit, ax=ax[0, 1], labelsize=labelsize, fontsize=fontsize)
    kratky_plot(data, guinier, filename=None, format=None, unit=unit, ax=ax[1, 0], labelsize=labelsize, fontsize=fontsize)
    density_plot(ift, filename=None, format=None, unit=unit, ax=ax[1, 1], labelsize=labelsize, fontsize=fontsize)
    if filename is not None:
        if format:
            fig.savefig(filename, format=format)
        else:
            fig.savefig(filename)
    return fig
