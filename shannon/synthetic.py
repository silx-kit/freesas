#!/usr/bin/env python3
# coding: utf-8
"""Synthetic SAXS curves with exactly known Dmax, Rg and I(0).

Ground truth is needed to compare Dmax estimators fairly:
 * a Dmax from GNOM or BIFT is itself an estimate, not a reference;
 * the training set of the neural network is unknown, so any SASBDB entry
   risks being contaminated (train/test leakage).

Analytical shapes let us sweep the parameters that actually matter: object
size, shape anisotropy, noise level and accessible q-range.

All lengths are in nm and q in 1/nm, matching the internal unit of FreeSAS
and the input grid of the neural network.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"

from collections import namedtuple

import numpy
from numpy import pi
from numpy.polynomial.legendre import leggauss
from numpy.polynomial import Polynomial

CURVE = namedtuple("CURVE", "q I sigma Dmax Rg I0 label")


def _phi(u):
    """Sphere form-factor amplitude, 3.j1(u)/u, normalised to 1 at u=0.

    sin(u) - u.cos(u) is a difference of two O(u) terms whose result is
    O(u^3/3): evaluating it directly loses all significant digits below
    u ~ 1e-3. Below u = 0.1 the truncated series is used instead, which is
    accurate to ~1e-14 there.
    """
    u = numpy.asarray(u, dtype=numpy.float64)
    out = numpy.empty_like(u)
    small = u < 0.1
    x2 = u[small] ** 2
    out[small] = 1.0 - x2 / 10.0 + x2**2 / 280.0 - x2**3 / 15120.0
    x = u[~small]
    out[~small] = 3.0 * (numpy.sin(x) - x * numpy.cos(x)) / x**3
    return out


def sphere(q, Dmax, I0=1.0):
    """Homogeneous sphere of diameter Dmax."""
    radius = 0.5 * Dmax
    return I0 * _phi(q * radius) ** 2


def spheroid(q, Dmax, epsilon=1.0, I0=1.0, npt=200):
    """Homogeneous spheroid, semi-axes (a, a, c) with c/a = epsilon.

    Dmax is the largest dimension, i.e. 2.max(a, c).
    epsilon > 1 is prolate (cigar), epsilon < 1 is oblate (pancake).
    """
    if epsilon >= 1.0:
        c = 0.5 * Dmax
        a = c / epsilon
    else:
        a = 0.5 * Dmax
        c = a * epsilon
    x, w = leggauss(npt)
    # map [-1, 1] -> [0, 1]
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    r = numpy.sqrt(a**2 * (1.0 - x**2) + c**2 * x**2)
    amp = _phi(q[:, None] * r[None, :])
    return I0 * (amp**2 * w[None, :]).sum(axis=1)


def core_shell_prolate(q, Dmax, epsilon=1.5, X=0.16, X_rho=-2.0, I0=1.0, npt=200):
    """Prolate core-shell spheroidal micelle, equations (S19)-(S21).

    This is the object class the paper was designed for: the electron-density
    contrast between core and shell creates fringes in I(q).

    :param Dmax: maximum micelle size, along the polar axis
    :param epsilon: core ellipticity c/a (> 1, prolate)
    :param X: shell thickness over maximum size, R_sh/Dmax
    :param X_rho: ratio of the electron-density differences,
                  d_rho(core-shell) / d_rho(shell-solution)
    """
    # prolate: epsilon_M > 1 so min(1, epsilon_M) = 1 and D_M = Dmax, X_eps = X
    X_eps = X
    core_axis = (1.0 - 2.0 * X_eps) / epsilon
    shell_axis = core_axis + 2.0 * X_eps
    coef_core = epsilon * core_axis**3 * X_rho
    coef_shell = shell_axis**2

    x, w = leggauss(npt)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    half_q = q[:, None] * 0.5 * Dmax
    u = half_q * numpy.sqrt(
        core_axis**2 * (1.0 - x**2)[None, :] + (1.0 - 2.0 * X_eps) ** 2 * (x**2)[None, :]
    )
    u1 = half_q * numpy.sqrt(
        shell_axis**2 * (1.0 - x**2)[None, :] + (x**2)[None, :]
    )
    amp = coef_core * _phi(u) + coef_shell * _phi(u1)
    raw = (amp**2 * w[None, :]).sum(axis=1)
    return I0 * raw / raw_at_zero(coef_core, coef_shell)


def raw_at_zero(coef_core, coef_shell):
    """Value of the core-shell intensity at q=0, for normalisation."""
    return (coef_core + coef_shell) ** 2


def parabola_pr(q, Dmax, I0=1.0, npt=2048):
    """Object whose pair-distance distribution is P(r) = -r(r - Dmax).

    Not a physical shape, but the exact sine transform is cheap and this is
    the case already used by FreeSAS' own test_bift.py.
    """
    r = numpy.linspace(0, Dmax, npt + 1)
    p = -r * (r - Dmax)
    sincqr = numpy.sinc(numpy.outer(q, r / pi))
    raw = (p * sincqr).sum(axis=-1)
    return I0 * raw / p.sum()


def exact_moments(func, Dmax, qmax_fit=None, **kwargs):
    """I(0) and Rg of a model, from the Guinier expansion at very low q.

    ln I = ln I0 - Rg^2 q^2 / 3 as q -> 0. Fitting a parabola in q^2 over a
    range with q.Rg << 1 gives both moments to machine-ish precision, and
    works for any of the shapes above without a dedicated formula.
    """
    if qmax_fit is None:
        qmax_fit = 0.3 / Dmax  # deep inside the Guinier region
    q = numpy.linspace(1e-6, qmax_fit, 400)
    intensity = func(q, Dmax, **kwargs)
    # Polynomial.fit rescales the domain to [-1, 1]: without it, fitting over
    # q^2 in [1e-12, qmax^2] is hopelessly ill-conditioned. Cubic in q^2
    # captures up to q^6 and leaves Rg accurate to ~1e-9 relative.
    series = Polynomial.fit(q**2, numpy.log(intensity), 3).convert()
    I0 = numpy.exp(series.coef[0])
    Rg = numpy.sqrt(-3.0 * series.coef[1])
    return I0, Rg


def add_noise(q, I, rel_error_at_qmax=0.05, seed=0, floor=0.0):
    """Counting-like noise whose relative amplitude grows as I decreases.

    sigma(q) = k.sqrt(I(q) + floor), with k set so that sigma/I reaches
    `rel_error_at_qmax` at the last point. This reproduces the essential
    feature of real SAXS data: the high-q tail is where the noise lives.

    :return: noisy intensity, sigma
    """
    rng = numpy.random.default_rng(seed)
    base = numpy.sqrt(numpy.abs(I) + floor)
    k = rel_error_at_qmax * abs(I[-1]) / base[-1]
    sigma = k * base
    return I + rng.normal(0.0, 1.0, I.shape) * sigma, sigma


def make_curve(
    shape="sphere",
    Dmax=10.0,
    qmin=0.05,
    qmax=4.0,
    npt=1024,
    rel_error_at_qmax=0.05,
    seed=0,
    **kwargs,
):
    """Build one synthetic dataset with its ground truth.

    :param shape: sphere | spheroid | core_shell | parabola
    :param Dmax: maximum size, in nm
    :param qmin, qmax, npt: sampling of the scattering vector, in 1/nm
    :param rel_error_at_qmax: noise level, relative error at the last point
    :param kwargs: passed to the shape function (epsilon, X, X_rho...)
    :return: CURVE
    """
    func = {
        "sphere": sphere,
        "spheroid": spheroid,
        "core_shell": core_shell_prolate,
        "parabola": parabola_pr,
    }[shape]
    q = numpy.linspace(qmin, qmax, npt)
    clean = func(q, Dmax, **kwargs)
    I0, Rg = exact_moments(func, Dmax, **kwargs)
    if rel_error_at_qmax > 0:
        noisy, sigma = add_noise(q, clean, rel_error_at_qmax, seed=seed)
    else:
        noisy, sigma = clean, numpy.full_like(clean, 1e-6 * clean.max())
    label = f"{shape} Dmax={Dmax:g}"
    if kwargs:
        label += " " + " ".join(f"{k}={v:g}" for k, v in kwargs.items())
    label += f" noise={rel_error_at_qmax:g}"
    return CURVE(q, noisy, sigma, Dmax, Rg, I0, label)


if __name__ == "__main__":
    # self-check: numeric moments against the analytical ones
    for Dmax in (5.0, 10.0, 20.0):
        radius = 0.5 * Dmax
        I0, Rg = exact_moments(sphere, Dmax)
        expected = numpy.sqrt(0.6) * radius
        print(f"sphere Dmax={Dmax:5g}  Rg={Rg:.6f} expected={expected:.6f} I0={I0:.6f}")
        for eps in (0.5, 1.0, 2.0):
            I0, Rg = exact_moments(spheroid, Dmax, epsilon=eps)
            if eps >= 1:
                c = radius
                a = c / eps
            else:
                a = radius
                c = a * eps
            expected = numpy.sqrt((2 * a**2 + c**2) / 5.0)
            print(
                f"  spheroid eps={eps:4g} Rg={Rg:.6f} expected={expected:.6f} I0={I0:.6f}"
            )
        I0, Rg = exact_moments(parabola_pr, Dmax)
        r = numpy.linspace(0, Dmax, 20001)
        p = -r * (r - Dmax)
        expected = numpy.sqrt(0.5 * numpy.trapezoid(p * r**2, r) / numpy.trapezoid(p, r))
        print(f"  parabola Rg={Rg:.6f} expected={expected:.6f} I0={I0:.6f}")
        I0, Rg = exact_moments(core_shell_prolate, Dmax, epsilon=1.5, X=0.16)
        print(f"  core_shell Rg={Rg:.6f} I0={I0:.6f}")
