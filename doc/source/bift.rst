Inverse Fourier Transform
=========================

Inverse Fourier transform is the (electron-) density as function of the pair-wise distance and
could theoretically be obtained from the Fourier inversion of the SAS curve.

This is an ill-posed problem due to the absence of phase in the scattering signal,
and the limited amount on information in SAS data.
This approach is doomed to fail as there is not unique solution but many solutions.

Various mathematical approaches have been developed with different additional constrains
like smoothness of the IFT curve, finite size of the particle ...

FreeSAS implements a Bayesian inversion method from Steen Hansen:
J. Appl. Cryst. (2000). 33, 1415-1421
and on the work of Jesse Hopkins in BioXTAS-RAW.

The result of `bift` is comparable but surely different from what `gnom` provides since the later is using Tikhonov regularization instead of bayesian inference.

How different has been measured on the complete BM29 archive: the two agree on
``Rg`` to about 1% but differ systematically by some 20% on ``Dmax``, and they
respond very differently to the error bars on the intensities. See
:doc:`freesas_vs_atsas`.
