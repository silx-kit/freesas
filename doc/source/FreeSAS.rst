Presentation of FreeSAS
=======================

FreeSAS is a Python package with small angles scattering tools in a MIT
type license. It provides:

* autorg: Automatique radius of giration assessement based on Guinier law
* cormap: Comparison of a set of (saxs) curves to decide if they are the same or not
* bift: Bayesian inverse fourier transform
* supcomb: overlay and averageing of dummy-atom models
* extract_ascii: a tool to extract `.dat` files from HDF5 files provided at ESRF-BM29  


Introduction
------------

FreeSAS has been write as a re-implementation of some ATSAS parts in
Python for a better integration in the BM29 ESRF beam-line processing
pipelines. It provides functions to read SAS data from pdb files and to
handle them. Parts of the code are written in Cython and parallelized to
speed-up the execution.

FreeSAS code is available on Github at https://github.com/kif/freesas .


Usage
-----

   
.. toctree::
   :maxdepth: 2

   cormap
   guinier
   bift
   dummy_atom_model
   quick_analysis

Project
-------

.. toctree::
   :maxdepth: 1

   installation	
   test
   coverage
   changelog
