Change-log
##########

FreeSAS v2026.10.0 07/10/2026
=============================
- BIFT: extra parameter to fit a constant background
- Compatibility with numpy 2.5
- Template-based output for all command line tools
- Numerical stability fixes in BIFT and in distance calculation
- Richer containers and improved plots, for BM29 among others
- Code quality: ruff linting, pathlib compatibility
- Release workflow: newer cibuildwheel, updated runners and GitHub actions
- e2e tests: replace PyPDF2 by pypdf
- Documentation: FreeSAS compared to ATSAS, contributor guidelines, and AGENTS.md
- Drops Python 3.10, supports Python 3.11-3.14

FreeSAS v2026.2.0 06/02/2026
============================
- Fix some tests
- Supports Python 3.10-3.14

FreeSAS v2024.9.0 11/09/2021
============================
- Rg+Dmax inferance from a Dense Neural Network (free_dnn)
- automatic packaging using github-actions
- Supports Python 3.9-3.12, including numpy2

FreeSAS v0.9.0 16/07/2021
=========================
- Command line tools upgrade
- end-to-end tests
- automatic packaging

FreeSAS v0.8.5 25/01/2021
=========================
- BIFT for invert Fourier Transform
- New implementation of auto-guinier
- BM29 HDF5-ascii extraction tool
- Continuous integration, doc and build

FreeSAS v0.7.0 31/08/2018
=========================
- Includes Correlation maps for curve equivalence assessment
- New packaging ala silx with debian 8 & 9 packages + manylinux wheels
- Few bug-fixes and always 3 scripts: autorg, supycomb and cormap

FreeSAS v0.6.0: 17/10/2017
==========================
- Radius of Gyration assessment based on Guinier law, with automatic determination of the linear region.
- Thanks to Martha Brennich & Jesse Hopkins for the code.

FreeSAS v0.4.0: 17/12/2015
==========================
 - First public version
 - built around bead-model alignment
