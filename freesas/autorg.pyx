# -*- coding: utf-8 -*-
"""
Loosely based on the autoRg implementation in BioXTAS RAW by J. Hopkins
"""
from __future__ import division, print_function
__authors__ = ["Martha Brennich", "Jerome Kieffer"]
__license__ = "MIT"
__copyright__ = "2017, EMBL"


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
RG_RESULT = namedtuple("RG_RESULT", ["Rg", "sigma_Rg", "I0", "sigma_I0", "start_point", "end_point", "quality", "aggregated"])

cimport numpy as cnumpy
import numpy as numpy 
from math import exp
from libc.math cimport sqrt, log 
from .isnan cimport isfinite 
from cython cimport floating
import logging
logger = logging.getLogger(__name__)


DTYPE = numpy.float64
ctypedef cnumpy.float64_t DTYPE_t

# Definition of a few constants
cdef: 
    DTYPE_t[::1] WEIGHTS
    int RATIO_INTENSITY = 10  # start with range from Imax -> Imax/10

qmaxrg_weight = 1.0
qminrg_weight = 0.1
rg_frac_err_weight = 1.0
i0_frac_err_weight = 1.0
r_sqr_weight = 4.0
reduced_chi_sqr_weight = 0.0
window_size_weight = 6.0
    
_weights = numpy.array([qmaxrg_weight, qminrg_weight, rg_frac_err_weight, 
                        i0_frac_err_weight, r_sqr_weight, reduced_chi_sqr_weight, 
                        window_size_weight])
WEIGHTS = numpy.ascontiguousarray(_weights / _weights.sum(), DTYPE)


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
    i_thres = i_max / RATIO_INTENSITY
    end = idx_out
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
             

def weightedlinFit(DTYPE_t[::1] datax, DTYPE_t[::1] datay, DTYPE_t[::1] weight, 
                   int data_start, int data_end, DTYPE_t[::1] results):
    """Calculates a fit to intercept-slope*x, weighted by w. 
        Input:
        x, y: The dataset to be fitted.
        w: The weight fot the individual points in x,y. Typically w would be 1/yerr**2.
        data_start and data_end: start and end for this fit
        Returns results: intercept, sigma_intercept, slope, sigma_slope
    """
    cdef: 
        int i, size
        DTYPE_t one_y, one_x, one_xy, one_xx, one_w, sigma_uxx, sigma_uxy, sigma_uyy
        DTYPE_t sigma_wy, sigma_wx, sigma_wxx, sigma_wxy, sigma_w, sigma_ux, sigma_uy
        DTYPE_t intercept, slope, xmean, ymean, ssxx, ssxy, ssyy, s, sigma_intercept, sigma_slope, xmean2
        DTYPE_t detA

    intercept = slope = xmean = ymean = ssxx = ssyy = ssxy = s = sigma_intercept = sigma_slope = 0.0 
    sigma_uxx = sigma_wy = sigma_wx = sigma_wxy = sigma_wxx = xmean2 = sigma_w = 0.0 
    sigma_uyy = sigma_uy = sigma_ux = sigma_uxy = 0.0

    assert len(results) >= 4, "There is enough room for storing results"
    size = data_end - data_start
    assert size > 2, "data range size should be >2"

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

    if abs(detA) > 1e-100:
        # x = [-sigma_wxx*sigma_wy + sigma_wx*sigma_wxy,-sigma_wx*sigma_wy+sigma_w*sigma_wxy]/detA
        intercept = (-sigma_wxx * sigma_wy + sigma_wx * sigma_wxy) / detA
        slope = (-sigma_wx * sigma_wy + sigma_w * sigma_wxy) / detA
        xmean = sigma_ux / size
        ymean = sigma_uy / size
        xmean2 = xmean * xmean
        ssxx = sigma_uxx - size * xmean2#*xmean
        ssyy = sigma_uyy - size * ymean * ymean
        s = sqrt((ssyy + slope * slope * ssxx) / (size - 2))
        sigma_intercept = sqrt(s * (1.0 / size + xmean2 / ssxx))
        sigma_slope = sqrt(s / ssxx)
        # xopt = (intercept, slope)
        # dx = (sigma_intercept, sigma_slope)
        # Returns results: intercept, sigma_intercept, slope, sigma_slope
        results[0] = intercept
        results[1] = sigma_intercept
        results[2] = slope
        results[3] = sigma_slope
    else:
        results[:] = 0


def calc_chi(DTYPE_t[::1] x, DTYPE_t[::1]y, DTYPE_t[::1] w,
             int start, int end, DTYPE_t offset, DTYPE_t slope,
             DTYPE_t[::1] fit_data):
    """Calculate the r_sqr, chi_sqr and reduced_chi_sqr to be saved in fit_data"""
    cdef: 
        int idx, size
        DTYPE_t value, sum_n, sum_y, sum_d, one_y, r_sqr, mean_y, value2, chi_sqr
    
    size = end - start
    assert size > 2, "more then 3 points in dataset"
    sum_n = 0.0
    sum_y = 0.0
    sum_d = 0.0
    chi_sqr = 0.0
    for idx in range(start, end):
        one_y = y[idx]
        value = (one_y - (offset - slope * x[idx]))
        value2 = value * value
        sum_n += value2
        sum_y += one_y
        chi_sqr += value2 * w[idx]
    mean_y = sum_y / size
    
    for idx in range(start, end):
        one_y = y[idx]
        value = one_y - mean_y
        sum_d = value * value
    r_sqr = 1.0 - sum_n / sum_d
    #r_sqr = 1 - diff2.sum()/((y-y.mean())*(y-y.mean())).sum()
    
    #if r_sqr > .15:
    #    chi_sqr = (diff2*yw).sum()
    reduced_chi_sqr = chi_sqr / (size - 2)
    
    fit_data[10] = r_sqr
    fit_data[11] = chi_sqr
    fit_data[12] = reduced_chi_sqr
    return r_sqr


def autoRg(sasm):
    """This function automatically calculates the radius of gyration and scattering intensity at zero angle
    from a given scattering profile. It roughly follows the method used by the autorg function in the atsas package
    Input:
    sasm: An array of q, I(q), dI(q)
    """
    cdef:
        DTYPE_t quality
        bint aggregated = 0
        cnumpy.ndarray qualities
        DTYPE_t[::1] q_ary, i_ary, sigma_ary, lgi_ary, q2_ary_ary, wg_ary, 
        DTYPE_t[::1] fit_result, fit_data
        int[::1] offsets, data_range
        int raw_size, currated_size, data_start, data_end
        int window_size, start, end
        list fit_list
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
    fit_result = numpy.zeros(4, dtype=DTYPE)
    
    currate_data(sasm, q_ary, i_ary, sigma_ary, q2_ary, lgi_ary, wg_ary, offsets, data_range)
    
    data_start, data_end, currated_size = data_range
    
    logger.debug("raw size: %s, currated size: %s start: %s end: %s", raw_size, currated_size, data_start, data_end)
    
    if (data_end - data_start) < 10:
        raise InsufficientDataError()
  

    #Pick a minimum fitting window size. 10 is consistent with atsas autorg.
    min_window = 10

    max_window = data_end - data_start

    fit_list = []

    # It is very time consuming to search every possible window size and every possible starting point.
    # Here we define a subset to search.
    tot_points = max_window
    window_step = tot_points // 10
    data_step = tot_points // 50

    if window_step == 0:
        window_step = 1
    if data_step == 0:
        data_step = 1

    # This function takes every window size in the window list, stepts it through the data range, and
    # fits it to get the RG and I0. If basic conditions are met, qmin*RG<1 and qmax*RG<1.35, and RG>0.1,
    # We keep the fit.
    for window_size in range(min_window, max_window + 1, window_step):
        for start in range(data_start, data_end - window_size, data_step):
            end = start + window_size
            # nota: fresh array each time
            fit_data = numpy.zeros(13, dtype=DTYPE)
            fit_data[0] = start
            fit_data[1] = window_size 
            fit_data[2] = q_ary[start]
            fit_data[3] = q_ary[end - 1]
            try: 
                weightedlinFit(q2_ary, lgi_ary, wg_ary, start, end, fit_result)
            except ValueError as VE:
                logger.error(VE)
                raise 
            except Exception as err:
                logger.error("An error occured: start=%s end=%s", start, end)
                raise 
            else:
                intercept, sigma_intercept, slope, sigma_slope = fit_result
                if (intercept == 0) and (slope == 0):
                    logger.error("Null determiant")
                    continue
                
                fit_data[4] = slope
                fit_data[5] = sigma_slope
                fit_data[6] = intercept
                fit_data[7] = sigma_intercept
                
                lower = q2_ary[start] * slope
                upper = q2_ary[start + window_size - 1] * slope

                fit_data[8] = lower 
                fit_data[9] = upper
                
                # check the validity of the model with some physics
                if (slope > 3e-5) and (lower < 0.33) and (upper < 0.6075) \
                        and (sigma_slope / slope <= 1):
                    r_sqr = calc_chi(q2_ary, lgi_ary, wg_ary, start, end, 
                                     intercept, slope, fit_data)
                    if r_sqr > .15:
                        fit_list.append(fit_data)
    
 
    if not fit_list:
        #Extreme cases: may need to relax the parameters.
        pass
    
    if fit_list:
        fit_array = numpy.vstack(fit_list)

        #Now we evaluate the quality of the fits based both on fitting data and on other criteria.


        max_window_real = float(max_window) #To ensure float division in Python 2

        #all_scores = []
        qmaxrg_score = 1-numpy.absolute((fit_array[:,9]-0.56)/0.56)
        qminrg_score = 1-fit_array[:,8]
        rg_frac_err_score = 1-fit_array[:,5]/fit_array[:,4]
        i0_frac_err_score = 1 - fit_array[:,7]/fit_array[:,6]
        r_sqr_score = fit_array[:,10]
        reduced_chi_sqr_score = 1/fit_array[:,12] #Not right
        window_size_score = fit_array[:,1]/max_window_real
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
                
                rg = sqrt(3.*fit_array[idx,4])
                dber = fit_array[:,5][qualities>qualities[idx]-.1].std()
                rger = 0.5*sqrt(3./rg)*dber
                i0 = exp(fit_array[idx,6])
                #i0 = fit_array[:,6][qualities>qualities[idx]-.1].mean()
                daer = fit_array[:,7][qualities>qualities[idx]-.1].std()
                i0er = i0*daer
                idx_min = int(fit_array[idx,0])
                idx_max = int(fit_array[idx,0]+fit_array[idx,1]-1)
                idx_min_corr = numpy.argmin(numpy.absolute(sasm[:,0] - fit_array[idx,3]))
                idx_max_corr = numpy.argmin(numpy.absolute(sasm[:,0] - fit_array[idx,4]))
            except:
                
                idx = qualities.argmax()
                rg = sqrt(3.*fit_array[idx,4])
                rger = 0.5*sqrt(3./rg)*fit_array[idx,5]
                i0 = exp(fit_array[idx,6])
                i0er = i0*fit_array[idx,7]
                idx_min = int(fit_array[idx,0])
                idx_max = int(fit_array[idx,0]+fit_array[idx,1]-1)
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

    idx_min = idx_min + data_start
    idx_max = idx_max + data_start

    #We could add another function here, if not good quality fits are found, either reiterate through the
    #the data and refit with looser criteria, or accept lower scores, possibly with larger error bars.

    return RG_RESULT(rg, rger, i0, i0er, idx_min, idx_max, quality, aggregated)
