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
from builtins import None, NotImplementedError

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

    def __init__(self, msg="Not enough data do determine Rg"):
        self.expression = msg
        self.message = msg


class NoGuinierRegionError(Error):
    def __init__(self, qRg_max=None):
        self.expression = ""
        self.message = "No Guinier region found with reasonnable qRg < %s"%qRg_max


import cython
cimport numpy as cnumpy
import numpy as numpy 
from libc.math cimport sqrt, log, fabs, exp, atanh, ceil, NAN
from .isnan cimport isfinite 
from cython cimport floating
import logging
logger = logging.getLogger(__name__)
from .collections import RG_RESULT, FIT_RESULT

DTYPE = numpy.float64
ctypedef double DTYPE_t

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


cdef inline DTYPE_t clamp(DTYPE_t x, 
                   DTYPE_t lower, 
                   DTYPE_t upper) nogil:
    "Equivalent of the numpy.clip"
    return min(upper, max(x, lower))


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
                      DTYPE_t intercept, 
                      DTYPE_t slope,
                      DTYPE_t[:, ::1] fit_mv, 
                      int position) nogil:
    """Calculate the rvalue, r², chi² and weigted RMSD to be saved in fit_data"""
    cdef: 
        int idx, size
        DTYPE_t residual, sum_n, sum_y, sum_d, one_y, R2, mean_y, residual2, sum_w
        DTYPE_t chi2, one_w, pred_y, sum_dev2
    
    size = end - start
    sum_n = 0.0
    sum_y = 0.0
    sum_d = 0.0
    sum_w = 0.0
    chi2 = 0.0
    sum_dev2 = 0.0
    for idx in range(start, end):
        one_y = y[idx]
        one_w = w[idx]
        pred_y = slope * x[idx] + intercept 
        residual = one_y - pred_y
        residual2 = residual * residual
        sum_n += residual2
        sum_y += one_y
        sum_w += one_w  
        chi2 += residual2 * one_w
        sum_dev2 += residual2 * one_w / (sum_y * sum_y)
    mean_y = sum_y / size
    for idx in range(start, end):
        one_y = y[idx]
        residual = one_y - mean_y
        sum_d += residual * residual
    R2 = 1.0 - sum_n / sum_d
    
    fit_mv[position, 0] = sqrt(R2)               #r_value
    fit_mv[position, 1] = R2                     #r²_value
    fit_mv[position, 2] = chi2                   # chi²
    fit_mv[position, 3] = sqrt(sum_dev2/sum_w)   # weigted RMSD
    
    return R2

cdef inline void guinier_space(int start, 
                               int stop, 
                               DTYPE_t[::1] q, 
                               DTYPE_t[::1] intensity, 
                               DTYPE_t[::1] sigma, 
                               DTYPE_t[::1] q2, 
                               DTYPE_t[::1] lnI, 
                               DTYPE_t[::1] I2_over_sigma2) nogil:
        "Initialize q², ln(I) and I/sigma array"
        cdef:
            int idx
            DTYPE_t one_q, one_i, one_sigma
    
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


def linear_fit(x, y, w, 
               int start=0, 
               int stop=0):
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
        stop = size + stop
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


cdef class AutoGuinier:
    "Calculate the radius of gyration based on Guinier's formula. This class holds all constants"
    cdef:
        readonly int min_size, storage_size
        readonly DTYPE_t Rg_min, qminrgmax, qmaxrgmax, relax, aggregation_threshold
    
    def __cinit__(self, 
                  int min_size=3, 
                  DTYPE_t Rg_min=1.0,
                  DTYPE_t qmin_rgmax=1.0,
                  DTYPE_t qmax_rgmax=1.3,
                  DTYPE_t relax=1.2,
                  DTYPE_t aggregation_threshold=0.1):
        
        """Constructor of the class with:
        
        :param min_size: minimal size for the Guinier region in number of points (3, so that one can fit a parabola)
        :param Rg_min: minimum acceptable value for the radius of gyration (1nm)
        :param qmin_Rgmax: maximum acceptable value for the begining of Guinier region.
        :param qmax_Rgmax: maximum acceptable value for the end of Guinier region.
        :param relax: relax the qmax_rgmax constrain by this value for reduced quality data      
        #weights: those parameters are used to measure the best region
        #:param 
        """
        self.storage_size = 20
        self.min_size = min_size
        self.Rg_min = Rg_min
        self.qminrgmax = qmin_rgmax
        self.qmaxrgmax = qmax_rgmax
        self.relax = relax
        self.aggregation_threshold = aggregation_threshold
    
    def __dealloc__(self):
        self.storage_size = 0
        self.min_size = 0
        self.Rg_min = 0.0
        self.qminrgmax = 0
        self.qmaxrgmax = 0
        self.relax = 0
        self.aggregation_threshold = 0
        
    @cython.profile(True)
    cpdef currate_data(self,
                       data, 
                       DTYPE_t[::1] q, 
                       DTYPE_t[::1] intensity,
                       DTYPE_t[::1] sigma,
                       DTYPE_t Rg_min=-1.0,
                       DTYPE_t qRg_max=-1.0,
                       DTYPE_t relax=-1.0):
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
        if qRg_max < 0.0:
            qRg_max = self.qmaxrgmax
        #qRg_max is now a hard threshold.
        if relax < 0.0:
            qRg_max *= self.relax
        else:
            qRg_max *= relax
        
        if Rg_min < 0.0:
            Rg_min = self.Rg_min
        
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
                if one_q*Rg_min < qRg_max:
                    end = idx
                else:
                    break
            else:
                q[idx] = NAN
                intensity[idx] = NAN
                sigma[idx] = NAN
                if (idx-start)<=self.min_size:
                    i_min = i_max = 0.0
                    start = - self.min_size -1
        return start, end+1

    def guinier_space(self, 
                       int start, 
                       int stop, 
                       DTYPE_t[::1] q, 
                       DTYPE_t[::1] intensity, 
                       DTYPE_t[::1] sigma, 
                       DTYPE_t[::1] q2, 
                       DTYPE_t[::1] lnI, 
                       DTYPE_t[::1] I2_over_sigma2):
        "Initialize q², ln(I) and I/sigma array"
        with nogil:
            guinier_space(start, stop, q, intensity, sigma, q2, lnI, I2_over_sigma2)

    @cython.profile(True)
    def many_fit(self, 
                 DTYPE_t[::1] q2_ary, 
                 DTYPE_t[::1] lnI_ary, 
                 DTYPE_t[::1] wg_ary,
                 int start,
                 int stop,  
                 DTYPE_t Rg_min=-1.0, 
                 DTYPE_t qRg_max=-1.0, 
                 DTYPE_t relax=-1.0):
        """Perform the linear regression for all reasonnably possible Guinier regions  
        
        :param q2_ary: array with q-squared (output of guinier_space)
        :param guinier_space: array with the log of the intensity (output of guinier_space)
        :param wg_ary: array with the weights (I²_over_sigma²)
        :param start: first valid point
        :param stop: last reasonnable valid point.
        :param Rg_min: Minimum acceptable radius of gyration (by default, the one from the class)
        :param qRg_max: End on the Guinier region (by default, the one from the class)
        :param relax: relax the qRg_max constrain by this value for reduced quality (by default, the one from the class)
        :return: an array of XXX-vector of floats, used as criteria to find the guinier region 
        
        0: Rg
        1: Rg_std
        2: I0
        3: I0_std
        4: start_point
        5: end_point
        6: quality
        7: agregated
        8: qRg_lower 
        9: qRg_upper
        10:14 slope, sigma_slope, intercept, sigma_intercept
        14:18 R, R², chi², RMDS
        18:20 q2Rg2_lower, q2Rg2_upper
        ...
        """
        cdef:
            int nb_fit, array_size, s, e, err
            DTYPE_t[:, ::1] result, tmp_mv
            DTYPE_t slope, sigma_slope, intercept, sigma_intercept, q2Rg2_lower, q2Rg2_upper
            DTYPE_t Rg2, Rg, qRg_upper, I0
            bint debug
        
        debug = logger.level<=logging.DEBUG
        nb_fit = 0
        array_size = 4096/(sizeof(DTYPE_t)*self.storage_size) #This correspond to one 4k page 
        result = numpy.zeros((array_size, self.storage_size), dtype=DTYPE)        

        with nogil:
            if qRg_max<0.0:
                qRg_max = self.qmaxrgmax
            #qRg_max is now a hard threshold.
            if relax<0.0:
                qRg_max *= self.relax
            else:
                qRg_max *= relax
            
            if Rg_min<0.0:
                Rg_min = self.Rg_min

            for s in range(start, stop-self.min_size):
                for e in range(s+self.min_size, stop):
                    # The 8 first parameters corresponds to the RG_RESULT
                    #result[nb_fit, 0] = Rg
                    #result[nb_fit, 1] = Rg_std
                    #result[nb_fit, 2] = I0
                    #result[nb_fit, 3] = I0_std
                    result[nb_fit, 4] = s
                    result[nb_fit, 5] = e 
                    #result[nb_fit, 6] = quality
                    #result[nb_fit, 7] = coef of the second order
                    #result[nb_fit, 8] = qRg_lower 
                    #result[nb_fit, 9] = qRg_upper
                    
                    # Coef 10-14 correspond to the linear regression
                    err = weighted_linear_fit(q2_ary, lnI_ary, wg_ary, s, e, result[:, 10:14], nb_fit)
                    if (err != 0):
                        if debug:
                            with gil:
                                logger.debug("position (%i,%i): Null determinant in linear regression", s, e)
                        result[nb_fit, :] = 0.0
                        continue                    
                    
                    slope = result[nb_fit, 10]
                    sigma_slope = result[nb_fit, 11] 
                    intercept = result[nb_fit, 12]
                    sigma_intercept = result[nb_fit, 13]
                    if slope >= 0:
                        result[nb_fit, :] = 0.0
                        if debug:
                            with gil:
                                logger.debug("position (%i,%i): Negative Rg²", s, e)
                        continue
                    # Coef 14-18 correspond to the assessement of the linear regression quality
                    if calc_chi(q2_ary, lnI_ary, wg_ary, s, e, 
                             intercept, slope, result[:, 14:18], nb_fit) <=0:
                        if debug:
                            with gil:
                                logger.debug("error in deviation calculation (%i, %i)", s, e)
                        # R² <0
                        result[nb_fit, :] = 0.0
                        continue
                    # Extract physical parameters:
                    Rg2 = -3.0*slope
                    result[nb_fit, 0] = Rg = sqrt(Rg2)
                    result[nb_fit, 1] = 0.5*sqrt(-3.0/slope)*sigma_slope # Rg_std
                    result[nb_fit, 2] = I0 = exp(intercept)
                    result[nb_fit, 3] = I0 * sigma_intercept #I0_std
                    
                    result[nb_fit, 18] = q2Rg2_lower = q2_ary[s] * Rg2
                    result[nb_fit, 19] = q2Rg2_upper = q2_ary[e-1] * Rg2
                    result[nb_fit, 8] = sqrt( q2Rg2_lower ) # qRg_lower
                    result[nb_fit, 9] = qRg_upper = sqrt( q2Rg2_upper )

                    if (Rg<Rg_min) or (qRg_upper>qRg_max):
                        if debug:
                            with gil:
                                logger.debug("Invalid qRg range (%i, %i) Rg %s > %s, qRg %s<%s", e, s, Rg, Rg_min, qRg_upper, qRg_max)
                        result[nb_fit, :] = 0.0
                        continue                        


                    #Claculate the descriptor for the quality
                    #result[nb_fit, 20] = result[nb_fit, 17]                          # 0 fit_score:  RMDS, normed
                    #result[nb_fit, 21] = 1.0 - <double>(e-s)/<double>(stop-start)    # 1 window_size_score: 
                    #result[nb_fit, 22] = q2Rg2_upper/self.q2maxrg2max                # 2 qmaxrg_score =  #quadratic penalty for qmax_Rg > 1.3
                    #result[nb_fit, 23] = q2Rg2_lower/self.q2minrg2max                # 3 qminrg_score =  #quadratic penalty for qmin_Rg > 1.0
                    #result[nb_fit, 24] = self.rg2_min/Rg2                            # 4 Rg_min score
                    #result[nb_fit, 25] = Rg_std/Rg                                   # 5 rg_error_score = 1.0 - sigma_Rg/Rg
                    #result[nb_fit, 26] = I0_std/I0                                   # 6 I0_error_score = 1.0 - sigma_I0/I0
                    
                    nb_fit +=1
                    if nb_fit >= array_size:
                        array_size *= 2
                        with gil:
                            tmp_mv = numpy.zeros((array_size, self.storage_size), dtype=DTYPE)
                            tmp_mv[:nb_fit, :] = result[:, :]
                            result = tmp_mv
        return numpy.asarray(result[:nb_fit])
    
    @cython.profile(True)
    def count_valid(self,
                    DTYPE_t[:, ::1] fit_result,
                    DTYPE_t qRg_max=-1.0, 
                    DTYPE_t relax=-1.0):
        """Count the number of valid Guinier intervals considering the qRg limit.  
        If no region is found, the constrain may be relaxed.
        
        Also searches for the maximum slope (absolute) value.

        :param fit_result: output of quality_fit: all possible Guinier reguions already fitted 
        :param qRg_max: upper bound of the Guinier region (1.3 is the standard value)
        :param relax: relax the qRg_max by this amount if no region are found.
        :return: number of regions, relaxed, actual qRg_max used, maximum slope
        """ 
        cdef:
            int i, size, cnt
            bint relaxed=0
            DTYPE_t aslope, aslope_max, qRg

        
        size = fit_result.shape[0]
        assert fit_result.shape[1] == self.storage_size, "fit_result shape is correct"
        with nogil:
            if qRg_max<0.0:
                qRg_max = self.qmaxrgmax
    
            #Start with strict mode, qRg_max being a soft-threshold
            aslope_max = 0.0
            cnt = 0
            for i in range(size):
                qRg = fit_result[i, 9]
                if qRg < qRg_max:
                    cnt += 1
                    aslope = -fit_result[i, 10] # slope is always negative by construction
                    if aslope>aslope_max:
                        aslope_max = aslope
            
            if cnt == 0:
                # There are not Guinier region whith reasonable qRg, let's relax the constrains
                relaxed = 1            
                if relax < 0.0:
                    qRg_max *= self.relax
                else:
                    qRg_max *= relax
                #qRg_max is now a hard threshold. 
                #Start with strict mode, qRg_max being a soft-threshold
                aslope_max = 0.0
                cnt = 0
                for i in range(size):
                    qRg = fit_result[i, 9]
                    if qRg < qRg_max:
                        cnt += 1
                        aslope = -fit_result[i, 10] # slope is always negative by construction
                        if aslope>aslope_max:
                            aslope_max = aslope
        return cnt, relaxed, qRg_max, aslope_max

    @cython.profile(True)
    cpdef (int, int) find_region(self, 
                                 DTYPE_t[:, ::1] fits, 
                                 DTYPE_t qRg_max=-1):
        """This function tries to extract the boundaries of the Guinier region from all fits.
        
        Each `start` and `stop` position is evaluated independantly, based on the average score of all contributing
        regions. Each region contributes with a weight calculated by:   
        * (q_max·Rg - q_min·Rg)/qRg_max --> in favor of large ranges
        * 1 / RMSD                      --> in favor of good quality data 
        
        For each start and end point, the contribution of all ranges are averaged out (using histograms-like techniques)
        The best solution is the (start,stop) couple position with the maximum score.
        
        To ensure start<stop, the stop is searched first (often less noisy) and the start is searched before stop.   
        
        :param fits: an array with all fits. Fits with q_max·Rg>qRg_max are ignored 
        :param qRg_max: the upper limit for searching the Guinier region. 
        :return 2-tuple with start-stop.
        """
        cdef:
            int start, stop, end, size, lower, upper, i
            cnumpy.int32_t[::1] unweigted_start, unweigted_stop
            DTYPE_t[::1] weigted_start,weigted_stop
            DTYPE_t max_weight, weight, qmin_Rg, qmax_Rg, RMSD 
        end = <int> numpy.max(fits[:,5])
        unweigted_start = numpy.zeros(end+1, dtype=numpy.int32)
        unweigted_stop = numpy.zeros(end+1, dtype=numpy.int32)
        weigted_start = numpy.zeros(end+1, dtype=DTYPE)
        weigted_stop = numpy.zeros(end+1, dtype=DTYPE)
        size = fits.shape[0]
        assert fits.shape[1] == self.storage_size, "size of fits matches"
        with nogil:
            if qRg_max<0.0:
                qRg_max = self.qmaxrgmax
            for i in range(size):
                qmax_Rg = fits[i, 9] #end of Guinier zone
                if qmax_Rg>qRg_max:
                    continue
                lower = <int> (fits[i, 4]) #lower index of the Guinier 
                upper = <int> (fits[i, 5]) #upperlower index of the Guinier
                qmin_Rg = fits[i, 8] #qRg at begining of Guinier zone
                RMSD = fits[i, 17]
                weight = (qmax_Rg - qmin_Rg)/RMSD # Empirically definied ... fits pretty well
                unweigted_start[lower] += 1
                unweigted_stop[upper] += 1
                weigted_start[lower] += weight
                weigted_stop[upper] += weight
            stop = 0
            max_weight = 0.0
            for i in range(end+1):
                if unweigted_stop[i]>0:
                    weight = weigted_stop[i]/unweigted_stop[i]
                    if weight>max_weight:
                        max_weight = weight
                        stop = i
            start=0
            max_weight = 0.0     
            for i in range(stop-self.min_size):
                if unweigted_start[i]>0:
                    weight = weigted_start[i]/unweigted_start[i]
                    if weight>max_weight:
                        max_weight = weight
                        start = i

        return start, stop
     
    @cython.profile(True)
    def slope_distribution(self,
                           DTYPE_t[:, ::1] fit_result,
                           int npt=1000,
                           DTYPE_t resolution=0.01,
                           DTYPE_t qRg_max=-1.0):
        """Find the most likely Guinier Region with updated parameters
        
        This function uses a 1D heat map of the |slope| values (with their errors) to assess the most likely Guinier region.
        This heat map is built of the sum of all gaussian curves, weighted by the q²Rg² range extension.
        
        Once the most likely |slope| has been found, the guinier region providing the nearest |slope| can be selected    
        
        :param fit_result: output of quality_fit: all possible Guinier reguions already fitted 
        :param npt: size of the distribution 
        :param resolution: The step size of the |slope|, from 0 to max(|slope|). Finer values provide better precision
        :param qRg_max: maximum allowed value for the upper limit of the Guinier region
        :return: the distribution of slopes.
        
        This function is finally not used.
        """
        cdef:
            int i, j, size
            DTYPE_t qRg, x, y, weight, aslope, sigma
            DTYPE_t[::1] distribution

        size = fit_result.shape[0]
        assert fit_result.shape[1] == self.storage_size, "fit_result shape is correct"

        distribution = numpy.zeros(npt, dtype=DTYPE)
        with nogil:
            for i in range(size):
                qRg = fit_result[i, 9]
                if qRg < qRg_max:
                    aslope = -fit_result[i, 10] # slope is always negative by contruction
                    sigma = fit_result[i, 11]
                    weight = (fit_result[i, 19]-fit_result[i, 18])/qRg_max
                    for j in range(npt):
                        x = j*resolution
                        y = weight*exp(-(x-aslope)**2/(2.0*sigma**2))/sigma
                        distribution[j] += y
        return numpy.asarray(distribution)
    
    @cython.profile(True)
    def average_values(self,
                       DTYPE_t[:, ::1] fit_result,
                       int start,
                       int stop
                       ):
        """Average out Rg and I0 and propagate errors from all sub-regions present in the main Guinier region
        
        :param: fit_result: output of quality_fit: all possible Guinier reguions already fitted 
        :paran start: the first index of the Guinier region
        :paran stop: the last index of the Guinier region
        :return: Rg_avg, Rg_std, I0_avg, I0_std, number of valid regions 
        """
        cdef:
            int i, good, size
            DTYPE_t wi, wr, swi, swr, srw, siw, Rg_avg, Rg_std, I0_avg, I0_std
        
        size = fit_result.shape[0]
        assert fit_result.shape[1] == self.storage_size, "fit_result shape is correct"
        with nogil:
            swi = swr = srw = siw = 0.0
            good = 0
            for i in range(size):
                if fit_result[i, 4]>=start and fit_result[i, 5]<=stop:
                    good +=1 
                    wr = 1/fit_result[i, 1]**2 # sigma_Rg
                    wi = 1/fit_result[i, 3]**2 # sigma_I0
                    swr += wr
                    swi += wi
                    srw += wr * fit_result[i,0] # Rg
                    siw += wi * fit_result[i,2] # I0
            Rg_avg = srw/swr
            I0_avg = siw/swi 
            swi = swr = srw = siw = 0.0
            good = 0
            for i in range(size):
                if fit_result[i, 4]>=start and fit_result[i, 5]<=stop:
                    good +=1 
                    wr = 1/fit_result[i, 1]**2 # sigma_Rg
                    wi = 1/fit_result[i, 3]**2 # sigma_I0
                    swr += wr
                    swi += wi
                    srw += wr * (fit_result[i,0]-Rg_avg)**2 # Rg
                    siw += wi * (fit_result[i,2]-I0_avg)**2 # I0
            Rg_std = sqrt(srw/swr)
            I0_std = sqrt(siw/swi)
        return Rg_avg, Rg_std, I0_avg, I0_std, good

    @cython.profile(True)
    def check_aggregation(self,
                          DTYPE_t[::1] q2, 
                          DTYPE_t[::1] lnI, 
                          DTYPE_t[::1] I2_over_sigma2, 
                          int start=0,
                          int end=0,
                          Rg=None,
                          threshold=None):
        """
        This function analyzes the curvature of a parabola fitted to the Guinier region 
        to check if the data indicate presence of larger aggragates
        
        A clear upwards curvature indicates aggregation.
        For convienniance , the polyfit from numpy is used which use the weights not squarred.
        
        :param q2: scattering vector squared
        :param lnI: logarithm if the intensity
        :param I2_over_sigma2: weight squared 
        :param start: the begining of the scan zone, argmax(I) is a good option, as early as possible
        :param end: the end of the scan zone, i.e. the end of the Guinier region
        :param Rg: the radius of gyration determined previously
        :param threshold: the value above which data are considered aggregated 
        :return: True if the protein is likely to be aggregated 
        if the threshold is None, return the value of the curvature*Rg**(-4)
        """
        cdef:
            DTYPE_t[::1] weight, coefs
            DTYPE_t value
        if end == 0:
            end=len(q2)
        else:
            assert len(q2)>=end, "q2 size matches"
        assert len(lnI)>=end, "lnI size matches"
        assert len(I2_over_sigma2)>=end, "I2_over_sigma2 size matches"
        weight = numpy.sqrt(I2_over_sigma2[start: end])
        try:
            coefs = numpy.polyfit(q2[start: end], lnI[start: end], 2, w=weight)
        except Exception as err:
            logger.error("Unable to fit parabola, %s: %s", type(err), err)
            return True
        if Rg:
            value = coefs[0]/Rg**4
        else:
            value = coefs[0]/(3*coefs[1])**2
    
        if threshold is False:
            return clamp(value, -1.0, 1.0)
        elif  threshold is None:
            return value>self.aggregation_threshold
        else:
            return value>threshold
    
    
    @cython.profile(True)
    cpdef DTYPE_t calc_quality(self, 
                               DTYPE_t Rg_avg, 
                               DTYPE_t Rg_std,
                               DTYPE_t qmin,
                               DTYPE_t qmax,
                               DTYPE_t aggregation,
                               DTYPE_t qRgmax = -1.0
                               ):
        """This function rates  the quality of the data.
        
        Unlike in `J. Appl. Cryst. (2007). 40, s223–s228`
        There I found no weight for how many consistent intervals were found.  
        
        :param Rg_avg: the average Rg found
        :param Rg_std: the standard deviation of Rg
        :param qmin: Scatteing vector of first point of the Guinier region
        :param qmax: Scatteing vector of last point of the Guinier region
        
        :return: quality indicator between 0 and 1
        """
        cdef:
            DTYPE_t quality
            DTYPE_t weight_Rg_dev, fit_Rg_dev, weight_qRgmax, fit_qRgmax, 
            DTYPE_t weight_drop, fit_drop, weight_aggregation, fit_aggregation, 
        
        if qRgmax <= 0:
            qRgmax = self.qmaxrgmax
        
        #0.58573097, 0.3070653 , 0.1795624 , 0.14731994, 2.6198585 
        
        #Quality form the fit of Rg
        weight_Rg_dev = 0.58573097
        fit_Rg_dev = 1.0 - clamp(Rg_std/Rg_avg, 0.0, 1.0)
         
        #Quality from the qRg max criteria  
        weight_qRgmax = 0.3070653
        fit_qRgmax = 1.0 - clamp(qmax*Rg_avg/qRgmax, 0.0, 1.0)
        
        #Quality from number of dropped points
        weight_drop = 0.1795624
        fit_drop = 1.0 - clamp(qmin/qmax, 0.0, 1.0)
        
        #quality from agregation value
        scale_aggregation = 2.5 
        weight_aggregation = 0.14731994
        fit_aggregation = 1.0 - clamp(scale_aggregation*fabs(aggregation), 0.0, 1.0)
        
        quality = fit_aggregation*weight_aggregation + \
                  fit_drop*weight_drop + \
                  fit_qRgmax*weight_qRgmax + \
                  fit_Rg_dev*weight_Rg_dev
        #as the sum of coef >1, can be larger than 1
        return clamp(quality, 0.0, 1.0)
    

    def fit(self, sasm):
        """This function automatically calculates the radius of gyration and scattering intensity at zero angle
        from a given scattering profile. It roughly follows the method used by the autorg function in the atsas package
        
        :param sasm: An array of q, I(q), dI(q)
        :return: RG_RESULT named tuple with the result of the fit
        """
        raise NotImplementedError()
    
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
                rger = 0.5 * sqrt(-3. / fit_array[idx, 4]) * fit_array[idx, 5]
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

    aggregated = guinier.check_aggregation(q2_ary, lgi_ary, wg_ary, data_start, offsets[idx_max], Rg=rg)
    return RG_RESULT(rg, rger, i0, i0er, offsets[idx_min], offsets[idx_max], quality, aggregated)

guinier = AutoGuinier() 


