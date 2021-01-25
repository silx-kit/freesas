Guinier fit
===========

André Guinier proved a small angle scattering curve can be approximated to 
$$
I = I_0 exp(-q²Rg²/3)
$$  
at low $q$, where Rg is the radius of giration of the scatterer.

The difficult part is to be able to find the Guinier-region, i.e. where this approximation is valid. 
FreeSAS implement 3 ways of selecting this region:

 - autogpa: Guinier peak analysis by Christopher D. Putnam
   J. Appl. Cryst. (2016). 49, 1412–1419
   Ti fits sqrt(q²Rg²)*exp(-q²Rg²/3)*I0/Rg to the curve I*q = f(q²)
   The Guinier region goes arbitrary from 0.5 to 1.3 q·Rg
 - autorg: Heavily inspired from Jesse Hopkins' BioXTAS RAW
   Journal of applied crystallography vol. 50,Pt 5 1545-1553.
 - auto-guinier: home brewed version: the main difference is that 
   one does not search for the "best region" but rather focuses on 
   the most likely start and end-points. 