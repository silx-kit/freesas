FreeSAS compared to ATSAS
=========================

This page summarises a cross-validation of FreeSAS against ATSAS carried out on
the complete BM29 analysis archive: **38 689 scattering curves recorded between
2018 and 2026**, each processed through both suites at both stages of the
standard workflow.

The purpose is not to declare a winner but to tell users what to expect when the
same data are analysed with either suite: where the two agree closely enough to
be interchangeable, where they do not, and why.

.. note::

   2019 is absent from the archive because the beamline was rebuilt that year.
   The rebuild also moved the accessible scattering vector range: before it
   ``q_min`` was about 0.035 nm\ :sup:`-1`, afterwards it ranges from 0.041 to
   0.078 nm\ :sup:`-1` depending on the detector distance in use.


What was compared
-----------------

Four independent routes to the same quantities, two per suite:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Stage
     - FreeSAS
     - ATSAS
     - Yields
   * - Guinier approximation
     - ``auto_guinier``
     - ``autorg``
     - ``Rg``, ``I(0)``
   * - Indirect Fourier transform
     - ``free_bift`` (BIFT)
     - ``datgnom`` (GNOM)
     - ``P(r)``, hence ``Rg``, ``I(0)``, ``Dmax``

All deviations below are expressed as FreeSAS relative to ATSAS, so a positive
bias means FreeSAS returns the larger value. Where a subset is described as
*both suites satisfied*, it means BIFT reached :math:`\chi^2 < 2` and GNOM
reported a ``Total Estimate`` above 0.7 — each suite's own verdict on its own
fit, never a criterion imposed from outside. 31 951 curves carry a ``Dmax``
from both suites, of which 12 714 satisfy both.


Radius of gyration: the two suites are mutually validated
---------------------------------------------------------

On the 12 714 curves where all four routes succeed:

.. list-table::
   :header-rows: 1
   :widths: 52 16 16 16

   * - Pairing
     - median \|dev\|
     - bias
     - within 5 %
   * - FreeSAS ``auto_guinier`` vs ATSAS ``autorg``
     - **0.96 %**
     - +0.20 %
     - 80.5 %
   * - ATSAS internal: ``autorg`` vs its own GNOM
     - 2.34 %
     - −0.48 %
     - 75.9 %
   * - FreeSAS BIFT vs ATSAS GNOM
     - 3.38 %
     - +2.63 %
     - 61.7 %
   * - FreeSAS internal: ``auto_guinier`` vs its own BIFT
     - 4.21 %
     - −3.07 %
     - 56.9 %

**The inter-suite disagreement is smaller than the intra-suite disagreement.**
FreeSAS and ATSAS place ``Rg`` within 0.96 % of each other at the Guinier stage,
while each suite's own Guinier and ``P(r)`` estimates differ by 2.34 % and
4.21 % respectively. The spread between methods exceeds the spread between
implementations, which is the strongest available evidence that both Guinier
implementations are correct.

``I(0)``, available from both ``P(r)`` routes, behaves likewise: 1.52 % median
disagreement.

The residual scatter is driven by noise and vanishes on good data — see
`Behaviour on noisy data`_, where the Guinier agreement reaches 0.40 % on the
cleanest fifth of the archive.


Maximum diameter: a systematic 20 % difference
----------------------------------------------

``Dmax`` behaves entirely differently. On the same 12 714 curves the two suites
differ by **20.27 %** in the median, with a bias of **+16.68 %**: BIFT
systematically returns the larger value.

The offset is reproducible across eight campaigns, a beamline rebuild and a
factor of two in ``q_min``, and its sign never changes:

.. list-table::
   :header-rows: 1
   :widths: 16 14 18 26 26

   * - Campaign
     - curves
     - ``q_min``
     - ``Rg`` (Guinier stage)
     - ``Dmax``, bias
   * - 2018
     - 4 331
     - 0.0349
     - 0.85 %
     - 22.63 %, +20.41 %
   * - 2020
     - 217
     - 0.0411
     - 1.08 %
     - 12.75 %, +10.24 %
   * - 2021
     - 1 509
     - 0.0440
     - 0.80 %
     - 19.48 %, +16.67 %
   * - 2022
     - 1 464
     - 0.0430
     - 1.14 %
     - 20.21 %, +16.90 %
   * - 2023
     - 1 695
     - 0.0629
     - 0.87 %
     - 17.03 %, +12.20 %
   * - 2024
     - 1 420
     - 0.0601
     - 1.08 %
     - 21.20 %, +13.21 %
   * - 2025
     - 1 581
     - 0.0852
     - 1.37 %
     - 21.10 %, +17.16 %
   * - 2026
     - 497
     - 0.0447
     - 0.95 %
     - 15.26 %, +11.01 %

The bias also varies in an orderly way with object size: about +23 % below
15 nm, decaying to +3.5 % at 30–50 nm, and **reversing to −18 % above 50 nm**,
where BIFT becomes the smaller of the two.

The reason for the contrast with ``Rg`` is structural rather than numerical.
``I(0)`` and ``Rg`` are **integral moments** of ``P(r)`` and are insensitive to
how its tail is treated, whereas ``Dmax`` is the **boundary of the support** —
precisely the quantity each regularisation scheme decides for itself.


Where the difference comes from
-------------------------------

Two experiments locate it.

The Guinier input is not the cause
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``datgnom`` requires an ``Rg`` on input, so GNOM can be run with the FreeSAS
Guinier result substituted for ``autorg``'s. On 4 000 curves the two input
values differ by 3.18 % in the median, and GNOM's output barely moves: 2.49 %
on ``Dmax``, 0.55 % on ``Rg``, 0.26 % on ``I(0)``. The propagation is damped,
:math:`\partial \ln D_{max} / \partial \ln R_g` being +0.50.

Decisively, on the same well-converged curves FreeSAS BIFT differs from pure
ATSAS GNOM by 19.90 % and from GNOM handed the FreeSAS ``Rg`` by 19.63 %.
**Feeding GNOM the FreeSAS Guinier result changes essentially nothing: the
difference lives in the transform.**

The regularisation principle is
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GNOM applies Tikhonov regularisation with the weight chosen from Svergun's
perceptual criteria, which are largely shape-based. BIFT infers both the
regularisation weight :math:`\alpha` and ``Dmax`` by maximising Bayesian
evidence, in which :math:`\sigma(q)` enters the likelihood.

That predicts a specific asymmetry, and rescaling the error bars while leaving
every intensity untouched tests it. On 40 real curves, each measured against its
own unperturbed result:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - :math:`\sigma` factor
     - BIFT ``Dmax``
     - BIFT :math:`\alpha`
     - GNOM ``Dmax``
   * - × 0.5
     - +0.34 %
     - 1.33×
     - **+0.00 %**
   * - × 2
     - −1.21 %
     - 0.66×
     - **+0.00 %**
   * - × 5
     - **−4.82 %**
     - 0.33×
     - **+0.00 %**

GNOM's ``Dmax`` does not move at all: for the purpose of terminating the
support it does not read the error bars. BIFT does — its :math:`\alpha` tracks
the claimed error level as the evidence rebalances :math:`\chi^2` against the
regularisation term, and ``Dmax`` follows. Told to distrust the data, BIFT
retreats to a more conservative support; GNOM returns the same answer.

.. warning::

   **BIFT's ``Dmax`` is only as trustworthy as the error bars supplied to it.**
   A pipeline that mis-scales :math:`\sigma(q)` — a forgotten normalisation, or
   errors propagated incorrectly through an averaging step — will shift BIFT's
   ``Dmax`` by several percent with no other warning. This is correct Bayesian
   behaviour rather than a defect, but it makes error propagation part of the
   ``Dmax`` uncertainty budget.


Robustness to damaged data
--------------------------

The same 40 curves were damaged in sixteen controlled ways and pushed through
both pipelines, 680 paired runs. Pooling by family of defect, median
\|``Dmax`` shift\| relative to each curve's own baseline:

.. list-table::
   :header-rows: 1
   :widths: 44 18 18 20

   * - Family of defect
     - BIFT
     - GNOM
     - more stable
   * - genuine extra noise
     - 2.91 %
     - 4.99 %
     - BIFT
   * - a species 3× larger added at 2–5 % of ``I(0)``
     - 0.98 %
     - 5.87 %
     - BIFT
   * - Guinier region removed
     - 1.32 %
     - 2.95 %
     - BIFT
   * - buffer over- or under-subtraction
     - 0.27 %
     - 0.76 %
     - BIFT
   * - error bars rescaled, data untouched
     - 2.02 %
     - 0.41 %
     - GNOM
   * - resolution removed (high ``q`` trimmed)
     - 1.24 %
     - 0.00 %
     - GNOM

BIFT is the more stable of the two on every family that alters the data itself.
GNOM wins only the two families that leave the low-``q`` information intact, and
it wins them because it does not use what was changed: discarding half the
measured ``q`` range moves its ``Dmax`` by exactly 0.00 %.

The widest margin in BIFT's favour is contamination. Adding a species three
times larger at just 2 % of ``I(0)`` shifts GNOM's ``Dmax`` by 5.38 % against
BIFT's 0.49 %, an elevenfold difference — relevant on real data, where a small
aggregated fraction is a routine hazard.


Behaviour on noisy data
-----------------------

The archive is far from clean: 31.5 % of curves carry more than 5 % non-positive
intensities, and 10.8 % have more than half their points below
:math:`I/\sigma = 1`.

Coverage
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 16 18

   * - Stage
     - all curves
     - worst quintile
     - best quintile
     - :math:`I/\sigma < 1` for > 50 % of points
   * - FreeSAS ``auto_guinier``
     - 99.9 %
     - 99.7 %
     - 100.0 %
     - 99.8 %
   * - ATSAS ``autorg``
     - 83.0 %
     - 65.2 %
     - 91.8 %
     - 41.7 %
   * - FreeSAS BIFT
     - 99.0 %
     - 96.4 %
     - 99.9 %
     - 96.0 %
   * - ATSAS GNOM
     - 83.0 %
     - 65.2 %
     - 91.8 %
     - 41.7 %

FreeSAS answers on essentially every curve; ATSAS declines 17 % of the archive
with *No Rg found* or *Data quality too low*, and 58 % of the worst tenth.
Quintiles are of the median :math:`I/\sigma` over the low-``q`` tenth of each
curve.

The 17 % refusal belongs entirely to ``autorg``: when ``datgnom`` is handed a
FreeSAS ``Rg``, **all 6 573 refused curves yield a** ``P(r)``. GNOM has no gate.
It does, however, have a verdict — the median ``Total Estimate`` on that
population is 0.448 against 0.673 on the accepted one, and only 15.5 % exceed
0.7. On those curves the two suites diverge by 38.79 % on ``Dmax`` and 39.85 %
on ``Rg``, against 20.27 % and 3.38 % on the consensus set. **The gate is
screening out data on which the two transforms cease to agree**, so bypassing
``autorg`` in production is not advisable.

How noise affects agreement
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - low-``q`` :math:`I/\sigma`
     - curves
     - ``Rg`` Guinier
     - ``Rg`` from ``P(r)``
     - ``Dmax``
   * - worst quintile
     - 1 561
     - 3.32 %
     - 7.93 %
     - 24.48 %
   * - best quintile
     - 1 790
     - **0.40 %**
     - **0.89 %**
     - 12.77 %
   * - improvement
     -
     - 8.3×
     - 8.9×
     - 1.9×

This separates the two problems cleanly. ``Rg`` disagreement is *noise-driven*:
it falls by a factor of nine as signal improves, until the two implementations
become interchangeable. ``Dmax`` disagreement is *method-driven*: it improves by
less than a factor of two and remains at 12.8 % on the best data available. No
amount of signal will make the two suites agree on ``Dmax``.

Screening in FreeSAS
~~~~~~~~~~~~~~~~~~~~

Returning an answer on poor data is only a problem if the user cannot tell. The
``quality`` field of ``RG_RESULT`` does not discriminate usefully: its median is
0.773 on the curves ATSAS accepts and 0.592 on those it rejects, and 94 % of the
rejected curves still score above 0.5. Any threshold on it either passes nearly
everything or rejects nearly everything.

What does discriminate is BIFT's :math:`\chi^2`: two thirds of the
ATSAS-rejected curves fail :math:`\chi^2 < 2`.

.. warning::

   Automated pipelines built on FreeSAS should screen on BIFT's
   :math:`\chi^2`, not on the ``quality`` field of the Guinier result.


Practical guidance
------------------

* **``Rg`` is safe to compare across suites.** 0.96 % median agreement at the
  Guinier stage, 0.40 % on good data, with no bias.
* **``Dmax`` is not.** Always name the software that produced it. A ``Dmax``
  aggregated from mixed sources carries a 20 % spread that no improvement in
  data quality removes.
* **Check the error scale before trusting a BIFT ``Dmax``.** ATSAS exposes a
  rescaling factor for exactly this purpose; BIFT has no equivalent guard, and
  it is the method that depends on :math:`\sigma(q)`.
* **Do not bypass ``autorg`` to gain coverage.** The curves it refuses are the
  ones on which the transforms disagree by 39 %.
* **Screen on** :math:`\chi^2`, **not on** ``quality``.
* For ``Dmax``, an interval reflecting the method spread is more honest than a
  point value quoted to three significant figures.


Scope and caveats
-----------------

* The 38 689 curves are not 38 689 independent measurements: 2 478 distinct
  sample prefixes appear in 2025 alone, several represented dozens of times.
  Medians and acceptance rates are robust to this, but no confidence interval
  should be derived from the sample size.
* The perturbation experiment uses 40 curves with a median baseline ``Dmax`` of
  about 10 nm. Medians over 40 curves are solid; the tails are not. Perturbation
  amplitudes were chosen rather than sampled from a real failure distribution,
  so the *ranking* of defect families is more meaningful than the absolute
  percentages.
* GNOM was always driven through ``datgnom``, which is the standard workflow but
  not the only way to run GNOM.
* Everything here concerns BM29 data reduced by the ESRF pipeline. The
  conclusions about ``Rg`` should carry over; those about coverage depend on the
  quality distribution of the archive.

The scripts producing every figure quoted on this page, together with the full
report they summarise, are kept outside the installable package in the
``shannon/`` directory of the source tree.

See :doc:`bift` for the FreeSAS inverse Fourier transform and :doc:`guinier`
for the Guinier analysis tools.
