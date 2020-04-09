# -*- coding: utf-8 -*-
##cython: boundscheck=False, wraparound=False, cdivision=True, embedsignature=True

__author__ = "Jerome Kieffer"
__license__ = "GPL"
__copyright__ = "2020, ESRF"
__date__ = "09/04/2020"


import cython
import numpy
from libc.math cimport sqrt, fabs

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def bift_inner_loop_cython(f, 
                           p, 
                           B, 
                           double alpha, 
                           int N, 
                           sum_dia,
                           int maxit=2000, 
                           int minit=100, 
                           double xprec=0.999, 
                           double omega=0.5,
                           double epsilon=1e-10):
    """
    Loop and seek for self consistence and smooth f
    
    Inner most function for BIFT ... most time is spent here.
    
    TODO: inline this function, f & p should be upgraded in place. Only dotsp should be returned
    
    :param f: guess density of size N+1, limits are null
    :param p: smoothed version of f over the 2 neighbours
    :param B: 2D array (N+1 x N+1) with 
    :param alpha: smoothing factor
    :param N: size of the problem
    :param maxit: Maximum number of iteration: 2000
    :param minit: Minimum number of iteration: 200
    :param xprec: Precision expected (->1)
    :param omega: Upgrade dumping factor, [0=no update, 1=full update]
    :param epsilon: The minimum density (except at boundaries)
    :return: updated version of f, p, sigma, dotsp, xprec
    """
    cdef:
        double dotsp, p_k, f_k, tmp_sum, fx, sigma_k, sum_dia_k, B_kk
        double sum_s2, sum_c2, sum_sc, s_k, c_k, sc_k
        int j, k
        double[::1] sigma_v, f_v, p_v, sum_dia_v
        double[:, ::1] B_v
    
    #Define starting conditions and loop variables
    dotsp = 0.0

    sigma = numpy.zeros((N+1), dtype=numpy.float64)
    
    #Create many views to work on arrays
    sigma_v = numpy.ascontiguousarray(sigma, dtype=numpy.float64)
    f_v = numpy.ascontiguousarray(f, dtype=numpy.float64)
    p_v = numpy.ascontiguousarray(p, dtype=numpy.float64)
    sum_dia_v = numpy.ascontiguousarray(sum_dia, dtype=numpy.float64)
    B_v = numpy.ascontiguousarray(B, dtype=numpy.float64)
    sum_c2 = sum_s2 = sum_sc = 0.0
    #Start loop
    for ite in range(maxit):
        if (ite > minit) and (dotsp > xprec):
            break

        #some kind of renormalization of the f & p vector to ensure positivity 
        for k in range(1, N):
            p_k = p_v[k]
            f_k = f_v[k]
            sigma_v[k] = abs(p_k + epsilon)            
            if p_k <= 0:
                p_v[k] = -p_k + epsilon
            if f_k <= 0:
                f_v[k] = -f_k + epsilon
            
        #Apply smoothness constraint: p is the smoothed version of f 
        for k in range(2, N-1):
            p_v[k] = 0.5 * (f_v[k-1] + f_v[k+1])
        p_v[0] = f_v[0] = 0.0
        p_v[1] = f_v[2] * 0.5
        p_v[N-1] = p_v[N-2] * 0.5 # is it p or f on the RHS?
        p_v[N] = f_v[N] = 0.0     # This enforces the boundary values to be null

        #Calculate the next correction
        for k in range(1, N):

            # a bunch of local variables:
            f_k = f_v[k]
            p_k = p_v[k]
            sigma_k = sigma_v[k]
            sum_dia_k = sum_dia_v[k]
            B_kk = B_v[k, k]
    
#             fsumi = numpy.dot(B[k, 1:N], f[1:N]) - B_kk*f_k
            tmp_sum = 0.0
            for j in range(1, N):
                tmp_sum += B_v[k, j]*f_v[j]
            tmp_sum -= B_v[k, k]*f_v[k]

            fx = (2.0*alpha* p_k/sigma_k + sum_dia_k - tmp_sum) / (2.0*alpha/sigma_k + B_kk)
            # Finally update the value 
            f_v[k] = f_k = (1.0-omega)*f_k + omega*fx

        # Calculate convergence
        sum_c2 = sum_sc = sum_s2 = 0.0
        for k in range(1, N):
            s_k = 2.0 * (p_v[k] - f_v[k]) / sigma_v[k]
            tmp_sum = 0.0
            for j in range(1, N):
                tmp_sum += B_v[k, j] * f_v[j]
            c_k = 2.0 * (tmp_sum - sum_dia_v[k])
            # There are 3 scalar products:
            sum_c2 += c_k * c_k
            sum_s2 += s_k * s_k
            sum_sc += s_k * c_k
        
        denom = sqrt(sum_s2*sum_c2)

#         gradsi = 2.0*(p[1:N] - f[1:N])/sigma[1:N]
#         gradci = 2.0*(numpy.dot(B[1:N,1:N],f[1:N]) - sum_dia[1:N])

#         wgrads2 = numpy.dot(gradsi,gradsi)
#         wgradc2 = numpy.dot(gradci, gradci)
#         denom = sqrt(wgrads2*wgradc2)
        if denom == 0:
            dotsp = 1.0
        else:
#             dotsp = numpy.dot(gradsi,gradci) / denom
            dotsp = sum_sc / denom
    return f, p, sigma, dotsp, xprec