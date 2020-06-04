# -*- coding: utf-8 -*-
#
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2020  European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Set of namedtuples defined a bit everywhere 
"""
__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__copyright__ = "2020 ESRF"

from collections import namedtuple
# Used in AutoRg
RG_RESULT = namedtuple("RG_RESULT", "Rg sigma_Rg I0 sigma_I0 start_point end_point quality aggregated")

def RG_RESULT_repr(self):
    return f"Rg={self.Rg:6.4f}(±{self.sigma_Rg:6.4}) I₀={self.I0:6.4f}(±{self.sigma_I0:6.4}) [{self.start_point}-{self.end_point}] {100.0*self.quality:5.2f}% {'aggregated' if self.aggregated>0.1 else ''}"
RG_RESULT.__repr__ = RG_RESULT_repr
FIT_RESULT = namedtuple("FIT_RESULT", "slope sigma_slope intercept sigma_intercept, R, R2, chi2, RMSD")

# Used in BIFT
RadiusKey = namedtuple("RadiusKey", "Dmax npt")
PriorKey = namedtuple("PriorKey", "type npt")
TransfoValue = namedtuple("TransfoValue", "transfo B sum_dia")
EvidenceKey = namedtuple("EvidenceKey", "Dmax alpha npt")
EvidenceResult = namedtuple("EvidenceResult", "evidence chi2r regularization radius density converged")
StatsResult = namedtuple("StatsResult", "radius density_avg density_std evidence_avg evidence_std Dmax_avg Dmax_std alpha_avg, alpha_std chi2r_avg chi2r_std regularization_avg regularization_std Rg_avg Rg_std I0_avg I0_std")

# Used in Cormap
GOF = namedtuple("GOF", ["n", "c", "P"])
