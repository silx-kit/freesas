# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, cdivision=True, embedsignature=True, language_level=3
# 
#    Project: freesas
#             https://github.com/kif/freesas
#
#    Copyright (C) 2017-2020  European Synchrotron Radiation Facility, Grenoble, France
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
Loosely based on the autoRg implementation in BioXTAS RAW by J. Hopkins
"""
__authors__ = ["Martha Brennich", "Jerome Kieffer"]
__license__ = "MIT"
__copyright__ = "2017 EMBL, 2020 ESRF"


class Error(Exception):
    """Base class for exceptions in this module."""
pass


class InsufficientDataError(Error):
    """Exception raised for errors in the input.
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self):
        self.expression = ""
        self.message = "Not enough data do determine Rg"

from collections import namedtuple
RG_RESULT = namedtuple("RG_RESULT", "Rg sigma_Rg I0 sigma_I0 start_point end_point quality aggregated")
FIT_RESULT = namedtuple("FIT_RESULT", "slope sigma_slope intercept sigma_intercept, R, R2, chi2, RMSD")

import cython
cimport numpy as cnumpy
import numpy as numpy 
from libc.math cimport sqrt, log, fabs, exp, atanh
from .isnan cimport isfinite 
from cython cimport floating
import logging
logger = logging.getLogger(__name__)


DTYPE = numpy.float64
ctypedef cnumpy.float64_t DTYPE_t

# Definition of a few constants
cdef: 
    DTYPE_t[::1] WEIGHTS
    int RATIO_INTENSITY = 10   # start with range from Imax -> Imax/10
    DTYPE_t RG_MIN = 0.0098    # minimum acceptable value
    DTYPE_t Q_MIN_RG_MAX = 1.0 # maximum acceptable values
    DTYPE_t Q_MAX_RG_MAX = 1.3 # maximum acceptable values
    # DERIVED VALUES
    DTYPE_t RG2_MIN = RG_MIN*RG_MIN
    DTYPE_t Q2_MIN_RG2_MAX = Q_MIN_RG_MAX*Q_MIN_RG_MAX
    DTYPE_t Q2_MAX_RG2_MAX = Q_MAX_RG_MAX*Q_MAX_RG_MAX
    
qmaxrg_weight = 1.0
qminrg_weight = 0.1
rg_frac_err_weight = 1.0
i0_frac_err_weight = 1.0
r_sqr_weight = 1000
reduced_chi_sqr_weight = 0.0
window_size_weight = 6.0
    
_weights = numpy.array([qmaxrg_weight, qminrg_weight, rg_frac_err_weight, 
                        i0_frac_err_weight, r_sqr_weight, reduced_chi_sqr_weight, 
                        window_size_weight])
WEIGHTS = numpy.ascontiguousarray(_weights / _weights.sum(), DTYPE)


cdef int weighted_linear_fit(DTYPE_t[::1] datax, 
                             DTYPE_t[::1] datay, 
                             DTYPE_t[::1] weight, 
                             int data_start, 
                             int data_end, 
                             DTYPE_t[:, ::1] fit_mv, 
                             int position) nogil:
    """Calculates a fit to intercept-slope*x, weighted by w. s
        Input:
        :param x, y: two dataset to be fitted.
        :param w: The weight fot the individual points in x,y. Typically w would be 1/yerr**2 or 1/(yerr²+a²xerr²).
        :param data_start: first valid point in arrays
        :param data_end: last valid point in arrays
        :param fit_mv: output array with result: has to be an array of nx4
        :param position: the index in fit_mv where result should be written  
        
        Returns results: intercept, sigma_intercept, slope, sigma_slope
                        in the array fit_mv at the given position 
    """
    cdef: 
        int i, size
        DTYPE_t one_y, one_x, one_xy, one_xx, one_w, sigma_uxx, sigma_uxy, sigma_uyy
        DTYPE_t sigma_wy, sigma_wx, sigma_wxx, sigma_wxy, sigma_w, sigma_ux, sigma_uy
        DTYPE_t intercept, slope, xmean, ymean, ssxx, ssxy, ssyy, s, sigma_intercept, sigma_slope, xmean2
        DTYPE_t detA, reduced_chi_sqr

    intercept = slope = xmean = ymean = ssxx = ssyy = ssxy = s = sigma_intercept = sigma_slope = 0.0 
    sigma_uxx = sigma_wy = sigma_wx = sigma_wxy = sigma_wxx = xmean2 = sigma_w = 0.0 
    sigma_uyy = sigma_uy = sigma_ux = sigma_uxy = 0.0

# There should be a python function performing the sanitization
#     assert fit_mv.shape[0] >= position, "There is enough room for storing results"
#     assert fit_mv.shape[1] >= 8, "There is enough room for storing results"
    size = data_end - data_start
#     assert size > 2, "data range size should be >2"

    for i in range(data_start, data_end):
        one_w = weight[i]
        one_y = datay[i]
        one_x = datax[i]
        sigma_w += one_w
        sigma_wy += one_w * one_y
        sigma_wx += one_w * one_x
        sigma_wxy += one_w * one_x * one_y
        sigma_wxx += one_w * one_x * one_x
        sigma_uxx += one_x * one_x
        sigma_uxy += one_x * one_y
        sigma_uy += one_y
        sigma_ux += one_x
        sigma_uyy += one_y * one_y
    
    detA = sigma_wx * sigma_wx - sigma_w * sigma_wxx

    if fabs(detA) > 1e-100:
        # x = [-sigma_wxx*sigma_wy + sigma_wx*sigma_wxy,-sigma_wx*sigma_wy+sigma_w*sigma_wxy]/detA
        intercept = (-sigma_wxx * sigma_wy + sigma_wx * sigma_wxy) / detA
        slope = (+sigma_wx * sigma_wy - sigma_w * sigma_wxy) / detA
        xmean = sigma_ux / size
        ymean = sigma_uy / size
        xmean2 = xmean * xmean
        ssxx = sigma_uxx - size * xmean2 # *xmean
        ssyy = sigma_uyy - size * ymean * ymean
        s = sqrt((ssyy + slope * slope * ssxx) / (size - 2))
        sigma_intercept = sqrt(s * (1.0 / size + xmean2 / ssxx))
        sigma_slope = sqrt(s / ssxx)
        # xopt = (intercept, slope)
        # dx = (sigma_intercept, sigma_slope)
        # Returns results: intercept, sigma_intercept, slope, sigma_slope
        fit_mv[position, 0] = slope
        fit_mv[position, 1] = sigma_slope
        fit_mv[position, 2] = intercept
        fit_mv[position, 3] = sigma_intercept
        return 0
    else:
        return -1

cdef DTYPE_t calc_chi(DTYPE_t[::1] x, 
                      DTYPE_t[::1] y, 
                      DTYPE_t[::1] w,
                      int start, 
                      int end, 
                      DTYPE_t offset, 
                      DTYPE_t slope,
                      DTYPE_t[:, ::1] fit_mv, 
                      int position) nogil:
    """Calculate the rvalue, r², chi² and reduced_chi² to be saved in fit_data"""
    cdef: 
        int idx, size
        DTYPE_t residual, sum_n, sum_y, sum_d, one_y, r_sqr, mean_y, residual2, sum_w
        DTYPE_t reduced_chi_sqr, chi_sqr, one_w
    

    size = end - start

    sum_n = 0.0
    sum_y = 0.0
    sum_d = 0.0
    sum_w = 0.0
    chi_sqr = 0.0
    
    for idx in range(start, end):
        one_y = y[idx]
        one_w = w[idx]
        residual = (one_y - (offset + slope * x[idx]))
        residual2 = residual * residual
        sum_n += residual2
        sum_y += one_y
        sum_w += one_w  
        chi_sqr += residual2 * one_w
    mean_y = sum_y / size
    
    for idx in range(start, end):
        one_y = y[idx]
        residual = one_y - mean_y
        sum_d += residual * residual

    r_sqr = 1.0 - sum_n / sum_d
    #r_sqr = 1 - diff2.sum()/((y-y.mean())*(y-y.mean())).sum()
    
    #if r_sqr > .15:
    #    chi_sqr = (diff2*yw).sum()
    reduced_chi_sqr = chi_sqr / (size - 2)
    
    fit_mv[position, 0] = sqrt(r_sqr)           #r_value
    fit_mv[position, 1] = r_sqr                 #r²_value
    fit_mv[position, 2] = chi_sqr               # chi²
    #fit_mv[position, 3] = reduced_chi_sqr      # reduces chi²
    fit_mv[position, 3] = sqrt(chi_sqr/sum_w)   # weigted RMSD
    return r_sqr


def linear_fit(x, y, w, 
               int start=0, 
               int stop=-1):
    """Wrapper for testing of weighted_linear_fit from Python
    
    :param x, y: The dataset to be fitted.
    :param w: The weight fot the individual points in x,y. Typically w would be 1/yerr² or 1/(yerr²+a²xerr²)
    :param start: start position in the array x, y, and w
    :param stop: end position in the array x, y, and w (end position excluded 
    :return: FIT_RESULT namedtuple with slope, sigma_slope intercept, sigma_intercept 
    """
    cdef:
        DTYPE_t[::1] datax, datay, weight 
        int size, er
        DTYPE_t[:, ::1] fit_mv
    
    datax = numpy.ascontiguousarray(x, dtype=DTYPE) 
    datay = numpy.ascontiguousarray(y, dtype=DTYPE)
    weight= numpy.ascontiguousarray(w, dtype=DTYPE)
    size = datax.shape[0]
    assert datay.shape[0] == size, "y size matches"
    assert weight.shape[0] == size
    fit_mv = numpy.zeros((1, 8), dtype=DTYPE)
    if stop<=0:
        stop = size - stop
    with nogil:
        er = weighted_linear_fit(datax, datay, weight, start, size, fit_mv[:, :4], 0)
        if er != 0:
            with gil:
                raise ArithmeticError("Null determinant in linear regression") 
        else:
            calc_chi(datax, datay, weight,
                     start, stop, fit_mv[0, 2], fit_mv[0, 0], 
                     fit_mv[:, 4:], 0)
        
    return FIT_RESULT(*fit_mv[0])    


cdef class AutoRG:
    "Calculate the radius of gyration based on Guinier's formula. This class holds all constants"
    cdef:
        readonly int min_size, weight_size, storage_size
        readonly DTYPE_t ratio_intensity, rg_min, rg2_min, qminrgmax, q2minrg2max, error_slope
        readonly DTYPE_t qmaxrgmax, q2maxrg2max
        public DTYPE_t[::1] weights
    
    def __cinit__(self, int min_size=3, 
                  DTYPE_t ratio_intensity=10.0,
                  DTYPE_t rg_min=1.0,
                  DTYPE_t qmin_rgmax=1.0,
                  DTYPE_t qmax_rgmax=1.3,
                  DTYPE_t error_slope=1.0):
        
        """Constructor of the class with:
        
        :param min_size: minimal size for the Guinier region in number of points (3, so that one can fit a parabola)
        :param ratio_intensity: search the Guinier region between Imax and Imax/ratio_intensity (10)
        :param rg_min: minimum acceptable value for the radius of gyration (1nm)
        :param qmin_rgmax: maximum acceptable value for the begining of Guinier region. This is the soft threshold (1.0 as default)
        :param qmax_rgmax: maximum acceptable value for the end of Guinier region. Note this is the hard threshold (1.3 as default)
        :param error_slope: discard any point with relative error on the slope (sigma_slope/slope)>=threhold, (1.0)
        weights: those parameters are used to measure the best region
        :param 
        """
        self.storage_size = 20
        self.weight_size = 7
        self.min_size = min_size
        self.ratio_intensity = ratio_intensity
        self.rg_min = rg_min
        self.rg2_min = rg_min*rg_min
        self.qminrgmax = qmin_rgmax
        self.q2minrg2max = qmin_rgmax*qmin_rgmax
        self.qmaxrgmax = qmax_rgmax
        self.q2maxrg2max = qmax_rgmax*qmax_rgmax
        self.error_slope = error_slope
        
        self.weights = numpy.empty(self.weight_size, dtype=DTYPE)
        self.weights
    
    def __dealloc__(self):
        self.weights = None
    
#     @cython.profile(True)
#     @cython.warn.undeclared(True)
#     @cython.warn.unused(True)
#     @cython.warn.unused_result(False)
#     @cython.warn.unused_arg(True)
    cpdef currate_data(self,
                       data, 
                       DTYPE_t[::1] q, 
                       DTYPE_t[::1] intensity,
                       DTYPE_t[::1] sigma):
        """Clean up the input (data, 2D array of q, i, sigma)
        
        It removed negatives q, intensity, sigmas and also NaNs and infinites
        q, intensity and sigma are ouput array. 
                
        :param data: input data, array of (q, I, sigma) of size N
        :param q: output array, to be filled with valid values
        :param intensity: output array, to be filled with valid values
        :param sigma: output array, to be filled with valid values
        :return: Begining and end of the valid region
        
        Nota: May work with more then 3 col and discard subsequent columns in data
        """
        cdef:
            int idx, size, start, end
            DTYPE_t one_q, one_i, one_sigma, i_max, i_min
            
        size = data.shape[0]
        q[:] = 0.0
        intensity[:] = 0.0
        sigma[:] = 0.0 
        # First pass: copy data and search for the max intensity
        start = self.min_size - 1 
        i_max = 0.0
        i_min = 0.0
        for idx in range(size):
            one_q = data[idx, 0]
            one_i = data[idx, 1]
            one_sigma = data[idx, 2]
            if isfinite(one_q) and one_q > 0.0 and \
               isfinite(one_i) and one_i > 0.0 and \
               isfinite(one_sigma) and one_sigma > 0.0:
                q[idx] = one_q
                intensity[idx] = one_i
                sigma[idx] = one_sigma
                if one_i > i_max:
                    i_min = i_max = one_i
                    start = idx
                if (idx-start)<=self.min_size:
                    i_min = min(i_min, one_i)
                if one_q*self.rg_min <self.qmaxrgmax:
                    end = idx
                else:
                    break
            else:
                q[idx] = numpy.NaN
                intensity[idx] = numpy.NaN
                sigma[idx] = numpy.NaN
                if (idx-start)<=self.min_size:
                    i_min = i_max = 0.0
                    start = - self.min_size -1
        #Extend the range to stlightly below the max value
        for idx in range(start, start - self.min_size, -1):
            if isfinite(q[idx]):
                start = idx;
            else:
                break
        return start, end+1

    cpdef guinier_space(self, 
                        int start, 
                        int stop, 
                        DTYPE_t[::1] q, 
                        DTYPE_t[::1] intensity, 
                        DTYPE_t[::1] sigma, 
                        DTYPE_t[::1] q2, 
                        DTYPE_t[::1] lnI, 
                        DTYPE_t[::1] I2_over_sigma2):
        "Initialize q², ln(I) and I/sigma array"
        cdef:
            int idx
            DTYPE_t one_q, one_i, one_sigma
    
        if 1: #with gil:with nogil:     if 1: #with gil:       
            q2[:] = 0.0
            lnI[:] = 0.0
            I2_over_sigma2[:] = 0.0
            
            for idx in range(start, stop):
                # populate the arrays for the fitting
                one_i = intensity[idx] 
                one_q = q[idx]
                q2[idx] = one_q * one_q
                lnI[idx] = log(one_i)
                one_sigma = sigma[idx]
                I2_over_sigma2[idx] = (one_i/one_sigma)**2
                 
    def quality_fit(self, sasm):
        """A function to calculate all the parameter vector (used to asses the quality) 
        
        :param sasm: An array of q, I(q), dI(q)
        :return: an array of 14-vector of floats, used as criteria to find the guinier region 
         
        0:4 start, window_size, q[start], q[end - 1], 
        4:8 slope, sigma_slope, intercept, sigma_intercept
        8:10 q²[start]Rg² <1²; q²[stop-1] * Rg²<1.3²
        10:12: Rg, I0
        12:16: R, R², chi², RMDS
        16: window_size_score = (e-s)/(stop) [0 - 1]
        17: Rg_score = Rg² - Rgmin²
        18: rmsd_score = 1.0 / rmsd
        19: R²_score = (R²-value) ** 4 
        20: qmaxrg_score =  #quadratic penalty for qmax_Rg > 1.3
        21: qminrg_score =  #quadratic penalty for qmin_Rg > 1 value is 1 at 0 and 0 at 1
        22: rg_error_score = 1.0 - sigma_slope/slope
        23: I0_error_score = 1.0 - sigma_intercept/intercept
        """
        cdef:
            int nb_fit, start, stop, array_size, s, e
            DTYPE_t[:, ::1] result 
            DTYPE_t[::1] q_ary, i_ary, sigma_ary, q2_ary, lgi_ary, wg_ary
            DTYPE_t slope, sigma_slope, intercept, sigma_intercept, q2Rg2_lower, q2Rg2_upper
            DTYPE_t Rg2, Rg

        nb_fit = 0
        array_size = 4096/(sizeof(DTYPE_t)*self.storage_size) #This correspond to one 4k page 
        result = numpy.zeros((array_size, self.storage_size), dtype=DTYPE)
        raw_size = sasm.shape[0]
        q_ary = numpy.empty(raw_size, dtype=DTYPE)
        i_ary = numpy.empty(raw_size, dtype=DTYPE)
        sigma_ary = numpy.empty(raw_size, dtype=DTYPE)
        q2_ary = numpy.empty(raw_size, dtype=DTYPE)
        lnI_ary = numpy.empty(raw_size, dtype=DTYPE)
        wg_ary = numpy.empty(raw_size, dtype=DTYPE)
        
        start, stop = autorg_instance.currate_data(sasm, q_ary, i_ary, sigma_ary)
        autorg_instance.guinier_space(start, stop, q_ary, i_ary,sigma_ary,
                                      q2_ary, lnI_ary, wg_ary)
        if 1: #with gil:with nogil:
            for s in range(start, stop-self.min_size):
                for e in range(s+self.min_size, stop):
                    result[nb_fit, 0] = s
                    result[nb_fit, 1] = e 
                    result[nb_fit, 2] = q_ary[s]
                    result[nb_fit, 3] = q_ary[e - 1]
                    
                    err = weighted_linear_fit(q2_ary, lnI_ary, wg_ary, s, e, result[:, 4:8], nb_fit)
                    if (err != 0):
                        if 1: #with gil:with gil:
                            logger.debug("position (%i,%i): Null determinant in linear regression", s, e)
                        result[nb_fit, :] = 0.0
                        continue
                    slope = result[nb_fit, 4]
                    sigma_slope = result[nb_fit, 5] 
                    intercept = result[nb_fit, 6]
                    sigma_intercept = result[nb_fit, 7]
                    if slope >= 0:
                        result[nb_fit, :] = 0.0
                        if 1: #with gil:with gil:
                            logger.debug("position (%i,%i): Negative Rg²", s, e)
                        continue
                    Rg2 = -3.0*slope
                    Rg = sqrt(Rg2)
                    q2Rg2_lower = q2_ary[s] * Rg2
                    q2Rg2_upper = q2_ary[e - 1] * Rg2
                    result[nb_fit, 8] = q2Rg2_lower 
                    result[nb_fit, 9] = q2Rg2_upper
                    result[nb_fit, 10] = Rg
                    result[nb_fit, 11] = exp(intercept)
                    calc_chi(q2_ary, lnI_ary, wg_ary, s, e, 
                             intercept, slope, result[:, 12:16], nb_fit)
                    if result[nb_fit, 13]<=0:  # R² <0
                        result[nb_fit, :] = 0.0
                        continue
                    #Calculate the descriptor ...
                    result[nb_fit, 16] = <double>(e-s)/<double>stop                  # 16: window_size_score = (e-s)/(stop-start) [0 - 1]
                    #result[nb_fit, 17] = -<double>s/<double>stop
                    #result[nb_fit, 17] = Rg2 - self.rg2_min                 # 17: Rg_score = Rg² - Rg_min²
                    #result[nb_fit, 18] = -result[nb_fit, 15]           # 18: rmsd_score = 1.0 / rmsd
                    result[nb_fit, 17] = (result[nb_fit, 13])**4            # 19: R²_score = (R²-value) ** 4 
                    result[nb_fit, 18] = 1.0 - (self.q2maxrg2max/q2Rg2_upper)   # 20: qmaxrg_score =  #quadratic penalty for qmax_Rg > 1.3
                    result[nb_fit, 19] = (Rg2/self.rg2_min) - 1.0
                    #result[nb_fit, 21] = - q2Rg2_lower/self.q2minrg2max # 21: qminrg_score =  #quadratic penalty for qmin_Rg > 1 value is 1 at 0 and 0 at 1
                    #result[nb_fit, 22] = 1.0 - sigma_slope/slope            # 22: rg_error_score = 1.0 - sigma_Rg/Rg
                    #result[nb_fit, 23] = 1.0 - sigma_intercept/intercept    # 23: I0_error_score = 1.0 - sigma_I0/I0
                    
                    nb_fit +=1
                    if nb_fit >= array_size:
                        array_size *= 2
                        if 1: #with gil:
                            tmp_mv = numpy.zeros((array_size, self.storage_size), dtype=DTYPE)
                            tmp_mv[:nb_fit, :] = result[:, :]
                            result = tmp_mv
        return numpy.asarray(result[:nb_fit])


#     def calc_
#     def fit_polynom(self, 
#                     DTYPE_t ):
    def fit(self, sasm):
        """This function automatically calculates the radius of gyration and scattering intensity at zero angle
        from a given scattering profile. It roughly follows the method used by the autorg function in the atsas package
        
        :param sasm: An array of q, I(q), dI(q)
        :return: RG_RESULT named tuple with the result of the fit
        """
        cdef:
            DTYPE_t quality, intercept, slope, sigma_slope, q2Rg2_lower, q2_Rg2_upper, r_sqr, rg2
            bint aggregated = 0
            cnumpy.ndarray qualities
            DTYPE_t[::1] q_ary, i_ary, sigma_ary, lgi_ary, q2_ary, wg_ary, 
            DTYPE_t[::1] fit_data
            cnumpy.int32_t[::1] offsets, data_range
            int raw_size, currated_size, data_start, data_end, data_step
            int min_window, max_window, window_size, window_step 
            int start, end, nb_fit, array_size, block_size=39 #page of 4k
            int idx_min, idx_max, idx, err
            DTYPE_t[:, ::1] fit_mv, tmp_mv
            cnumpy.ndarray[DTYPE_t, ndim=2] fit_array
            
        raw_size = sasm.shape[0]
        q_ary = numpy.empty(raw_size, dtype=DTYPE)
        i_ary = numpy.empty(raw_size, dtype=DTYPE)
        sigma_ary = numpy.empty(raw_size, dtype=DTYPE)
        q2_ary = numpy.empty(raw_size, dtype=DTYPE)
        lgi_ary = numpy.empty(raw_size, dtype=DTYPE)
        wg_ary = numpy.empty(raw_size, dtype=DTYPE)
        offsets = numpy.empty(raw_size, dtype=numpy.int32)
        array_size = block_size
        fit_mv = numpy.zeros((array_size, self.storage_size), dtype=DTYPE)
    
        data_start, data_end = self.currate_data(sasm, q_ary, i_ary, sigma_ary)
        
#                                        #q2_ary, lgi_ary, wg_ary, offsets, data_range)
#         with nogil:
#             # Pick a minimum fitting window size. 10 is consistent with atsas autorg.   
#             min_window = self.min_size
#             max_window = data_end - data_start
#         
#             if (data_end - data_start) < min_window:
#                 with gil:
#                     raise InsufficientDataError("Length of linear region, from %s to %s, is less than %s points long"%(data_end, data_start, min_window))
#       
#         
#         # This function takes every window size in the window list, stepts it through 
#         # the data range, and fits it to get the RG and I0. If basic conditions are 
#         # met, qmin*RG<1 and qmax*RG<1.35, and RG>0.1,
#         # We keep the fit.
#     #     with nogil:
#             # It is very time consuming to search every possible window size and every 
#             # possible starting point.
#             # Here we define a subset to search.
#             window_step = max(max_window // 10, 1)
#             data_step = max(max_window // 50, 1)
#             for window_size from min_window <= window_size < max_window + 1 by window_step:
#                 #for start in range(data_start, data_end - window_size, data_step):
#                 for start from data_start <= start < data_end - window_size by data_step:
#                     end = start + window_size
#                     #logger.debug("Fitting: %s , %s ", start,end)
#                     fit_mv[nb_fit, 0] = start
#                     fit_mv[nb_fit, 1] = window_size 
#                     fit_mv[nb_fit, 2] = q_ary[start]
#                     fit_mv[nb_fit, 3] = q_ary[end - 1]
#     
#                     err = weighted_linear_fit(q2_ary, lgi_ary, wg_ary, start, end, fit_mv[:, 4:8], nb_fit)
#     
#                     if (err != 0):
#                         with gil:
#                             logger.error("Null determinant in linear regression")
#                             continue
#         
#                     slope = fit_mv[nb_fit, 4] 
#                     rg2 = -slope
#                     sigma_slope = fit_mv[nb_fit, 5] 
#                     intercept = fit_mv[nb_fit, 6]
#                     q2Rg2_lower = q2_ary[start] * rg2
#                     q2_Rg2_upper = q2_ary[start + window_size - 1] * rg2
#     
#                     fit_mv[nb_fit, 8] = q2Rg2_lower 
#                     fit_mv[nb_fit, 9] = q2_Rg2_upper
#                     # check the validity of the model with some physics
#                     # i. e qmin*RG<1 and qmax*RG<1.35, and RG>0.1,
#                     if (rg2 > self.rg2_min) and (q2_Rg2_upper < self.q2rg2max) and (sigma_slope / rg2 <= self.error_slope):
#                         r_sqr = calc_chi(q2_ary, lgi_ary, wg_ary, start, end, 
#                                          intercept, slope, fit_mv[:, 10:14], nb_fit)
#                         if r_sqr > .15:
#                             nb_fit += 1
#                             # Allocate some more memory if needed
#                             if nb_fit >= array_size:
#                                 array_size *= 2
#                                 with gil:
#                                     tmp_mv = numpy.zeros((array_size, self.storage_size), dtype=DTYPE)
#                                     tmp_mv[:nb_fit, :] = fit_mv[:, :]
#                                     fit_mv = tmp_mv
#                         else:
#                             #reset data
#                             fit_mv[nb_fit, :] = 0.0
#                     else:
#                             fit_mv[nb_fit, :] = 0.0
#         
#         logger.debug("Number of valid fits: %s ", nb_fit)
#                         
#         if nb_fit == 0:
#             #Extreme cases: may need to relax the parameters.
#             pass
#         
#         if nb_fit > 0:
#             fit_array = numpy.asarray(fit_mv)[:nb_fit, :]
#     
#             #Now we evaluate the quality of the fits based both on fitting data and on other criteria.
#     
#             #all_scores = []
#             qmaxrg_score = 1.0 - numpy.absolute((fit_array[:, 9]-0.56)/0.56)
#             qminrg_score = 1.0 - fit_array[:, 8]
#             rg_frac_err_score = 1.0 - fit_array[:, 5]/fit_array[:, 4]
#             i0_frac_err_score = 1.0 - fit_array[:, 7]/fit_array[:, 6]
#             r_sqr_score = fit_array[:, 10]**4
#             reduced_chi_sqr_score = 1.0 / fit_array[:,12] #Not right
#             window_size_score = fit_array[:, 1] / max_window #float dividion forced by fit_array dtype 
#             scores = numpy.array([qmaxrg_score, qminrg_score, rg_frac_err_score, i0_frac_err_score, r_sqr_score,
#                                   reduced_chi_sqr_score, window_size_score])
#             qualities = numpy.dot(WEIGHTS, scores)
#            
#             #I have picked an aribtrary threshold here. Not sure if 0.6 is a good qualities cutoff or not.
#             if qualities.max() > 0:# 0.5:
#                 # idx = qualities.argmax()
#                 # rg = fit_array[idx,4]
#                 # rger1 = fit_array[idx,5]
#                 # i0 = fit_array[idx,6]
#                 # i0er = fit_array[idx,7]
#                 # idx_min = fit_array[idx,0]
#                 # idx_max = fit_array[idx,0]+fit_array[idx,1]
#     
#                 # try:
#                 #     #This adds in uncertainty based on the standard deviation of values with high qualities scores
#                 #     #again, the range of the qualities score is fairly aribtrary. It should be refined against real
#                 #     #data at some point.
#                 #     rger2 = fit_array[:,4][qualities>qualities[idx]-.1].std()
#                 #     rger = rger1 + rger2
#                 # except:
#                 #     rger = rger1
#     
#                 try:
#                     idx = qualities.argmax()
#                     #rg = fit_array[:,4][qualities>qualities[idx]-.1].mean()
#                     
#                     rg = sqrt(-3. * fit_array[idx, 4])
#                     dber = fit_array[:, 5][qualities > qualities[idx] - .1].std()
#                     rger = 0.5 * sqrt(3. / rg) * dber
#                     i0 = exp(fit_array[idx, 6])
#                     #i0 = fit_array[:,6][qualities>qualities[idx]-.1].mean()
#                     daer = fit_array[:, 7][qualities > qualities[idx] - .1].std()
#                     i0er = i0 * daer
#                     idx_min = int(fit_array[idx, 0])
#                     idx_max = int(fit_array[idx, 0] + fit_array[idx, 1] - 1.0)
#     #                 idx_min_corr = numpy.argmin(numpy.absolute(sasm[:, 0] - fit_array[idx, 3]))
#     #                 idx_max_corr = numpy.argmin(numpy.absolute(sasm[:, 0] - fit_array[idx, 4]))
#                 except:
#                     
#                     idx = qualities.argmax()
#                     rg = sqrt(-3. * fit_array[idx, 4])
#                     rger = 0.5 * sqrt(3. / rg) * fit_array[idx, 5]
#                     i0 = exp(fit_array[idx, 6])
#                     i0er = i0 * fit_array[idx, 7]
#                     idx_min = int(fit_array[idx, 0])
#                     idx_max = int(fit_array[idx, 0] + fit_array[idx, 1] - 1.0)
#                 quality = qualities[idx]
#             else:
#               
#                 rg = -1
#                 rger = -1
#                 i0 = -1
#                 i0er = -1
#                 idx_min = -1
#                 idx_max = -1
#                 quality = 0
#     
#         else:
#            
#             rg = -1
#             rger = -1
#             i0 = -1
#             i0er = -1
#             idx_min = -1
#             idx_max = -1
#             quality = 0
#             all_scores = []
#     
#         # managed by offsets
#         # idx_min = idx_min + data_start
#         # idx_max = idx_max + data_start
#     
#         #We could add another function here, if not good quality fits are found, either reiterate through the
#         #the data and refit with looser criteria, or accept lower scores, possibly with larger error bars.
#     
#         return RG_RESULT(rg, rger, i0, i0er, offsets[idx_min], offsets[idx_max], quality, aggregated)
#     __call__ = fit
    
################################################################################
# Old implementation from Matha    
################################################################################
def currate_data(floating[:, :] data, 
                 DTYPE_t[::1] q, 
                 DTYPE_t[::1] intensity,
                 DTYPE_t[::1] sigma,
                 DTYPE_t[::1] q2,
                 DTYPE_t[::1] log_intensity,
                 DTYPE_t[::1] weights,
                 int[::1] offsets,
                 int[::1] data_range):
    """Clean up the input (data, 2D array of q, i, sigma)
    
    It removed negatives q, intensity, sigmas and also NaNs and infinites
    q, intensity and sigma are ouput array. 
    
    we need also x: q*q, y: log I and w: (err/i)**(-2) 

    
    :param data: input data, array of q,i,sigma of size N
    :param q: output array, to be filled with valid values
    :param intensity: output array, to be filled with valid values
    :param sigma: output array, to be filled with valid values
    :param offsets: output array, provide the index in the input array for an 
                   index in the output array
    :return: the number of valid points in the array n <=N
    """
    cdef:
        int idx_in, idx_out, size_in, size_out, start, end, idx
        DTYPE_t one_q, one_i, one_sigma, i_max, i_thres, tmp
        
    size_in = data.shape[0]
    
    # it may work with more then 3 col and discard subsequent columns
    # assert data.shape[1] == 3, "data has 3 columns" 
    assert q.size >= size_in, "size of q_array is valid"
    assert intensity.size >= size_in, "size of intensity array is valid"
    assert sigma.size >= size_in, "size of sigma array is valid"
    assert q2.size >= size_in, "size of q2 array is valid"
    assert log_intensity.size >= size_in, "size of log_intensity array is valid"
    assert weights.size >= size_in, "size of weights array is valid" 
    assert offsets.size >= size_in, "size of offsets array is valid"
    assert data_range.size >= 3, "data range holds enough space"
    
    # For safety: memset the arrays
    q[:] = 0.0
    intensity[:] = 0.0
    sigma[:] = 0.0
    q2[:] = 0.0
    log_intensity[:] = 0.0
    weights[:] = 0.0
    offsets[:] = 0
    data_range[:] = 0 
    start = 0  
    idx_out = 0
    i_max = 0.0
    for idx_in in range(size_in):
        one_q = data[idx_in, 0]
        one_i = data[idx_in, 1]
        one_sigma = data[idx_in, 2]
        if isfinite(one_q) and one_q > 0.0 and \
           isfinite(one_i) and one_i > 0.0 and \
           isfinite(one_sigma) and one_sigma > 0.0:
            q[idx_out] = one_q
            intensity[idx_out] = one_i
            sigma[idx_out] = one_sigma
            offsets[idx_out] = idx_in
            if one_i > i_max:
                i_max = one_i
                start = idx_out
            idx_out += 1
            
    # Second pass: focus on the valid region and prepare the 3 other arrays
    
    end = idx_out
    if end > start + 2:
        i_thres = (i_max + data[start + 1, 1] + data[start + 2, 1]) / (3 * RATIO_INTENSITY)
    else:
        i_thres = i_max / (RATIO_INTENSITY)
    
    for idx in range(start, idx_out):
        one_i = intensity[idx] 
        if one_i < i_thres:
            end = idx
            break
        else:
            # populate the arrays for the fitting  
            one_q = q[idx]
            q2[idx] = one_q * one_q
            log_intensity[idx] = log(one_i)
            # w = (i/sigma)**2
            one_sigma = sigma[idx]
            tmp = one_i / one_sigma
            weights[idx] = tmp * tmp
        
    data_range[0] = start        
    data_range[1] = end
    data_range[2] = idx_out   

def autoRg(sasm):
    """This function automatically calculates the radius of gyration and scattering intensity at zero angle
    from a given scattering profile. It roughly follows the method used by the autorg function in the atsas package
    Input:
    sasm: An array of q, I(q), dI(q)
    """
    cdef:
        DTYPE_t quality, intercept, slope, sigma_slope, lower, upper, r_sqr
        bint aggregated = 0
        cnumpy.ndarray qualities
        DTYPE_t[::1] q_ary, i_ary, sigma_ary, lgi_ary, q2_ary, wg_ary, 
        DTYPE_t[::1] fit_data
        int[::1] offsets, data_range
        int raw_size, currated_size, data_start, data_end, data_step
        int min_window, max_window, window_size, window_step 
        int start, end, nb_fit, array_size, block_size=39 #page of 4k
        int idx_min, idx_max, idx, err
        DTYPE_t[:, ::1] fit_mv, tmp_mv
        cnumpy.ndarray[DTYPE_t, ndim=2] fit_array
        
    raw_size = len(sasm)
    q_ary = numpy.empty(raw_size, dtype=DTYPE)
    i_ary = numpy.empty(raw_size, dtype=DTYPE)
    sigma_ary = numpy.empty(raw_size, dtype=DTYPE)
    q2_ary = numpy.empty(raw_size, dtype=DTYPE)
    lgi_ary = numpy.empty(raw_size, dtype=DTYPE)
    wg_ary = numpy.empty(raw_size, dtype=DTYPE)
    offsets = numpy.empty(raw_size, dtype=numpy.int32)
    data_range = numpy.zeros(3, dtype=numpy.int32)
    
    currate_data(sasm, q_ary, i_ary, sigma_ary, q2_ary, lgi_ary, wg_ary, offsets, data_range)
    
    data_start, data_end, currated_size = data_range
    
    logger.debug("raw size: %s, currated size: %s start: %s end: %s", raw_size, currated_size, data_start, data_end)
   
    if (data_end - data_start) < 10:
        raise InsufficientDataError()
  
    # Pick a minimum fitting window size. 10 is consistent with atsas autorg.
    min_window = 10
    max_window = data_end - data_start

    # It is very time consuming to search every possible window size and every 
    # possible starting point.
    # Here we define a subset to search.
    window_step = max_window // 10
    data_step = max_window // 50

    if window_step == 0:
        window_step = 1
    if data_step == 0:
        data_step = 1

    array_size = block_size
    nb_fit = 0
    fit_mv = numpy.zeros((array_size, 14), dtype=DTYPE)
    # This function takes every window size in the window list, stepts it through 
    # the data range, and fits it to get the RG and I0. If basic conditions are 
    # met, qmin*RG<1 and qmax*RG<1.35, and RG>0.1,
    # We keep the fit.
    with nogil:
        #for window_size in range(min_window, max_window + 1 , window_step):
        for window_size from min_window <= window_size < max_window + 1 by window_step:
            #for start in range(data_start, data_end - window_size, data_step):
            for start from data_start <= start < data_end - window_size by data_step:
                end = start + window_size
                #logger.debug("Fitting: %s , %s ", start,end)
                fit_mv[nb_fit, 0] = start
                fit_mv[nb_fit, 1] = window_size 
                fit_mv[nb_fit, 2] = q_ary[start]
                fit_mv[nb_fit, 3] = q_ary[end - 1]

                err = weighted_linear_fit(q2_ary, lgi_ary, wg_ary, start, end, fit_mv[:,4:8], nb_fit)

                if (err<0):
                    with gil:
                        logger.debug("Null determiant")
                        fit_mv[nb_fit, :] = 0.0
                        continue
    
                slope = -fit_mv[nb_fit, 4] 
                sigma_slope = fit_mv[nb_fit, 5] 
                intercept = fit_mv[nb_fit, 6]
                lower = q2_ary[start] * slope
                upper = q2_ary[start + window_size - 1] * slope

                fit_mv[nb_fit, 8] = lower 
                fit_mv[nb_fit, 9] = upper
                
                # check the validity of the model with some physics
                # i. e qmin*RG<1 and qmax*RG<1.35, and RG>0.1,
#                 if (slope >= 1/3.0) and (lower <= 1**2/3) and (upper <= 1.35**2/3) \
#                         and (sigma_slope / slope <= 1):
                if (slope > 3e-5) and (lower < 0.33) and (upper < 0.6075) \
                        and (sigma_slope / slope <= 1):
                    r_sqr = calc_chi(q2_ary, lgi_ary, wg_ary, start, end, 
                                     intercept, -slope, fit_mv[:,10:14], nb_fit)
                    if r_sqr > .15:
                        nb_fit += 1
                        if nb_fit >= array_size:
                            array_size *= 2
                            with gil:
                                tmp_mv = numpy.zeros((array_size, 14), dtype=DTYPE)
                                tmp_mv[:nb_fit, :] = fit_mv[:, :]
                                fit_mv = tmp_mv
                    else:
#                         with gil:
#                             logger.debug("(%i, %i) Invalid R²=%s", start, end, r_sqr)
                        #reset data
                        fit_mv[nb_fit, :] = 0.0
                else:
#                     with gil:
#                         logger.debug("(%i, %i) Invalid Rg slope: %s, lower: %s, upper %s sigma: %s\n %s", start, end, slope>3e-5, lower>0.33, upper<0.6075, 
#                                      sigma_slope / slope<1, numpy.asarray(fit_mv[nb_fit]))
                    fit_mv[nb_fit, :] = 0.0
    

    logger.debug("Number of valid fits: %s ", nb_fit)
                    
    if nb_fit == 0:
        #Extreme cases: may need to relax the parameters.
        pass
    
    if nb_fit > 0:
        fit_array = numpy.asarray(fit_mv)[:nb_fit, :]
        #Now we evaluate the quality of the fits based both on fitting data and on other criteria.

        #all_scores = []
        qmaxrg_score = 1.0 - numpy.absolute((fit_array[:, 9]-0.56)/0.56)
        qminrg_score = 1.0 - fit_array[:, 8]
        rg_frac_err_score = 1.0 - fit_array[:, 5]/fit_array[:, 4]
        i0_frac_err_score = 1.0 - fit_array[:, 7]/fit_array[:, 6]
        r_sqr_score = fit_array[:, 11]**4
        reduced_chi_sqr_score = 1.0 / fit_array[:,12] #Not right
        window_size_score = fit_array[:, 1] / max_window #float dividion forced by fit_array dtype 
        scores = numpy.array([qmaxrg_score, qminrg_score, rg_frac_err_score, i0_frac_err_score, r_sqr_score,
                              reduced_chi_sqr_score, window_size_score])
        qualities = numpy.dot(WEIGHTS, scores)
        #I have picked an aribtrary threshold here. Not sure if 0.6 is a good qualities cutoff or not.
        if qualities.max() > 0:# 0.5:
            # idx = qualities.argmax()
            # rg = fit_array[idx,4]
            # rger1 = fit_array[idx,5]
            # i0 = fit_array[idx,6]
            # i0er = fit_array[idx,7]
            # idx_min = fit_array[idx,0]
            # idx_max = fit_array[idx,0]+fit_array[idx,1]

            # try:
            #     #This adds in uncertainty based on the standard deviation of values with high qualities scores
            #     #again, the range of the qualities score is fairly aribtrary. It should be refined against real
            #     #data at some point.
            #     rger2 = fit_array[:,4][qualities>qualities[idx]-.1].std()
            #     rger = rger1 + rger2
            # except:
            #     rger = rger1

            try:
                idx = qualities.argmax()
                #rg = fit_array[:,4][qualities>qualities[idx]-.1].mean()
                
                rg = sqrt(-3. * fit_array[idx, 4])
                dber = fit_array[:, 5][qualities > qualities[idx] - .1].std()
                rger = 0.5 * sqrt(3. / rg) * dber
                i0 = exp(fit_array[idx, 6])
                #i0 = fit_array[:,6][qualities>qualities[idx]-.1].mean()
                daer = fit_array[:, 7][qualities > qualities[idx] - .1].std()
                i0er = i0 * daer
                idx_min = int(fit_array[idx, 0])
                idx_max = int(fit_array[idx, 0] + fit_array[idx, 1] - 1.0)
                idx_min_corr = numpy.argmin(numpy.absolute(sasm[:, 0] - fit_array[idx, 3]))
                idx_max_corr = numpy.argmin(numpy.absolute(sasm[:, 0] - fit_array[idx, 4]))
            except:
                idx = qualities.argmax()
                rg = sqrt(-3. * fit_array[idx, 4])
                rger = 0.5 * sqrt(3. / rg) * fit_array[idx, 5]
                i0 = exp(fit_array[idx, 6])
                i0er = i0 * fit_array[idx, 7]
                idx_min = int(fit_array[idx, 0])
                idx_max = int(fit_array[idx, 0] + fit_array[idx, 1] - 1.0)
            quality = qualities[idx]
        else:
          
            rg = -1
            rger = -1
            i0 = -1
            i0er = -1
            idx_min = -1
            idx_max = -1
            quality = 0

    else:
       
        rg = -1
        rger = -1
        i0 = -1
        i0er = -1
        idx_min = -1
        idx_max = -1
        quality = 0
        all_scores = []

    # managed by offsets
    # idx_min = idx_min + data_start
    # idx_max = idx_max + data_start

    #We could add another function here, if not good quality fits are found, either reiterate through the
    #the data and refit with looser criteria, or accept lower scores, possibly with larger error bars.
    
    return RG_RESULT(rg, rger, i0, i0er, offsets[idx_min], offsets[idx_max], quality, aggregated)

autorg_instance = AutoRG() 
autoRg_ng = autorg_instance.fit

