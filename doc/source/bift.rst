Inverse Fourier Trasform
========================

Inverse Fourier transform is the (electron-) density as function of the pair-wise distance and
could theoritically be obtained from the Fourier inversion of the SAS curve. 

This is an ill-posed problem due to the absence of phase in the scattering signal, 
and the limited amount on information in SAS data. 
This approach is doomed to fail as there is not unique solution but many solutions.

Various mathematical approches have been developped with different additional constrains
like smoothness of the IFT curve, finite size of the particle ... 

FreeSAS implements a Bayesian investion method from Steen Hansen: 
J. Appl. Cryst. (2000). 33, 1415-1421
and on the work of Jesse Hopkins in BioXTAS-RAW.

The result of `bift` is comparable but surely different from what `gnom` provides since the later is using Tikonov 
regularisations instead of bayesian inference.