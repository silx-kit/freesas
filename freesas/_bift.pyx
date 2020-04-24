# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, initializedcheck=False, 
#cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#cython: boundscheck=False, wraparound=False, cdivision=True, 
"""
Bayesian Inverse Fourier Transform

This code is the implementation of 
Steen Hansen J. Appl. Cryst. (2000). 33, 1415-1421

Based on the BIFT from Jesse Hopkins, available at:
https://sourceforge.net/p/bioxtasraw/git/ci/master/tree/bioxtasraw/BIFT.py

This is a major rewrite in Cython 
"""

__authors__ = ["Jerome Kieffer", "Jesse Hopkins"]
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "24/04/2020"


import cython
import numpy
from libc.math cimport sqrt, fabs, pi, sin, log, exp
from collections import namedtuple
import logging
logger = logging.getLogger(__name__)

RadiusKey = namedtuple("RadiusKey", "Dmax npt")
PriorKey = namedtuple("PriorKey", "I0 Dmax npt")
EvidenceKey = namedtuple("EvidenceKey", "Dmax alpha npt")
EvidenceResult = namedtuple("EvidenceResult", "evidence chi2 regularization radius density")
StatsResult = namedtuple("StatsResult", "radius density_avg density_std evidence_avg evidence_std Dmax_avg Dmax_std alpha_avg, alpha_std chi2_avg chi2_std regularization_avg regularization_std Rg_avg Rg_std I0_avg I0_std")


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef distribution_sphere(double I0, 
                          double Dmax,
                          int npt):
    """Creates the initial P(r) function for the prior as a sphere.

    Formula from Svergun & Koch 2003:

    p(r) = 4*pi*Dmax**3/24*r**2*(1-1.5*r/Dmax+0.5*(r/Dmax)**3)

    Normalized so that the area is equal to the measured I(0). Integrate p(r):

    I(0) = integral of p(r) from 0 to Dmax
         = (4*pi*D^3/24)**2

    So the normalized p(r) is:

    p(r) = r**2*(1-1.5*r/Dmax+0.5*(r/Dmax)**3) * I0/(4*pi*Dmax**3/24)

    To match convention with old RAW BIFT, we also carry around an extra factor
    of 4*pi*Delta r, so the normalization becomes:

    p(r) = r**2*(1-1.5*r/Dmax+0.5*(r/Dmax)**3) * I0/(Dmax**3/(24*Delta_r))
    
    :param I0: forward scattering intensity
    :param npt: number of points in the real space (-1)
    :param Dmax: Diameter of the the object, here a sphere
    :return: the density p(r)
    """
    cdef:
        int j
        double norm, delta_r, r 
        double[::1] p_r
        
    norm = I0 / (4.0 * pi * Dmax**3 / 24.0)
    delta_r = Dmax / npt

    p_r = numpy.empty(npt+1, dtype=numpy.float64)
    
    for j in range(npt+1):
        r = j * delta_r
        p_r[j] = norm * r**2 * (1.0 - 1.5*(r/Dmax) + 0.5*(r/Dmax)**3)
    # p = p * I0/(4*pi*Dmax**3/24.)
    # p = p * I0/(Dmax**3/(24.*(r[1]-r[0])))   #Which normalization should I use? I'm not sure either agrees with what Hansen does.

    return numpy.asarray(p_r)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef distribution_parabola(double I0, 
                            double Dmax,
                            int npt):
    """Creates the initial p(r) function for the prior as a parabola.

    p(r) = r * (Dmax - r)
    
    I0 = 4*pi * Dmax³/6 
    
    :param I0: forward scattering intensity
    :param npt: number of points in the real space (-1)
    :param Dmax: Diameter of the the object, here a sphere
    :return: the density p(r)
    """
    cdef:
        int j
        double norm, delta_r, r 
        double[::1] p_r
        
    norm = I0 / (4.0 * pi * Dmax**3 / 6.0)
    delta_r = Dmax / npt

    p_r = numpy.empty(npt+1, dtype=numpy.float64)
    
    for j in range(npt+1):
        r = j * delta_r
        p_r[j] = norm * r * (Dmax - r) 

    return numpy.asarray(p_r)


cdef class BIFT:
    """Bayesian Inverse Fourier Transform
    
    :param q: scattering vector in 1/nm or 1A, the unit of q imposes the one on Dmax, r, ...
    :param I: Scattering intensity I(q) 
    :param I_std: error on the intensity estimation    """
    cdef:
        readonly int size, high_start, high_stop
        readonly double I0_guess, delta_q, Dmax_guess, alpha_max
        readonly double[::1] q, intensity, variance
        readonly dict prior_cache, evidence_cache, radius_cache, transfo_cache
     
    def __cinit__(self, q, I, I_std):
        """Constructor of the Cython class
        :param q: scattering vector in 1/nm or 1A, the unit of q imposes the one on Dmax, r, ...
        :param I: Scattering intensity I(q) 
        :param I_std: error on the intensity estimation
        """
        self.size = q.shape[0]
        assert self.size == I.shape[0], "Intensity array matches in size"
        assert self.size == I_std.shape[0], "Error array matches in size"
        self.q = numpy.ascontiguousarray(q, dtype=numpy.float64)
        self.intensity = numpy.ascontiguousarray(I, dtype=numpy.float64)
        self.variance = numpy.ascontiguousarray(I_std**2, dtype=numpy.float64)
        self.delta_q = (q[self.size-1]-q[0]) / (q.size-1)
        #We define a region of high signal where the noise is expected to be minimal:
        self.I0_guess = numpy.max(I)  # might be replaced with replaced with data from the Guinier fit  
        self.high_start = numpy.argmax(I) # Might be replaced by the guinier region
        self.high_stop = self.high_start + numpy.where(I[self.high_start:]<self.I0_guess/2.)[0][0]
        self.Dmax_guess = 0.0
        self.alpha_max = 0.0
        self.prior_cache = {}
        self.evidence_cache = {}
        self.radius_cache = {}
        self.transfo_cache = {}
         
    def __dealloc__(self):
        "This is the destructor of the class: free the memory and empty all caches"
        self.reset()
        self.q = self.intensity = self.variance = None

    def reset(self):
        "rest all caches"
        for cache  in (self.prior_cache,  self.evidence_cache, self.radius_cache, self.transfo_cache):
            if cache is not None:
                for key in list(cache.keys()):
                    cache.pop(key)

    def set_Guinier(self, guinier_fit, factor=3.0):
        """Set some starting point from Guinier fit like:
        
        Dmax = 3 Rg
        I0 for prior density modeling
        Guinier region for high signal with low noise section 
        
        :param guinier_fit: RG_RESULT instance from autorg fit
        :param factor: guess the Dmax =  factor * Rg
        :return: guessed Dmax
        """
        self.I0_guess = guinier_fit.I0
        self.high_start = guinier_fit.start_point
        self.high_stop = guinier_fit.end_point
        self.Dmax_guess = guinier_fit.Rg * factor
    
    def guess_alpha_max(self, int npt):
        """This is to define the maximum realistic alpha value to scan for.
         
        idea: limit case is alpha·S0 = chi2/2
        
        so calculate the prior p, the regularization factor associated, ...
        
        :param npt: number of points in the real space (-1)
        """
        cdef:
            double[::1] density, smooth
            double[:, ::1] tranfo
            double regularization, chi2
        if self.Dmax_guess<=0.0:
            raise RuntimeError("Please initialize with Guinier fit data using set_Guinier")
        density = self.prior_distribution(self.I0_guess, self.Dmax_guess, npt)
        smooth = numpy.zeros(npt+1, numpy.float64)
        self.smooth_density(density, smooth, npt)
        regularization = self.calc_regularization(density, smooth, density, npt) # eq19
        transfo = self.get_trans_matrix(self.Dmax_guess, npt)
        chi2 = self.calc_chi2(transfo, density, npt)*self.size
        return 0.5*chi2/regularization
    
    def get_best(self):
        """Return the most probable configuration found so far
        """
        best_evidence = numpy.finfo(numpy.float64).min
        best_key = None 
        for key, value in self.evidence_cache.items():
            if value.evidence>best_evidence:
                best_key = key
                best_evidence = value.evidence
        return best_key, self.evidence_cache.get(best_key)
    
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def prior_distribution(self,
                           double I0, 
                           double Dmax, 
                           int npt, 
                           dist_type='sphere'):
        """Calculate the prior distribution for the bayesian analysis
        
        Implements memoizing
        
        :param I0: forward scattering intensity, often approximated by th maximum intensity
        :param Dmax: Largest dimention of the object
        :param npt: number of points in the real space (-1)
        :param dist_type: str, for now only "sphere" is acceptable
        :return: the distance distribution function p(r) where r = numpy.linspace(0, Dmax, npt+1) 
        """
        if dist_type != 'sphere':
            raise RuntimeError("Only 'sphere' is accepted for dist_type")
        # manage the cache
        key = PriorKey(I0, Dmax, npt)
        if key in self.prior_cache:
            value = self.prior_cache[key]
        else:
            value = self.prior_cache[key] = distribution_sphere(I0, Dmax, npt)
        return value
    
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def radius(self, Dmax, npt):
        """Calculate the radius array with memoizing
        
        Rationnal: the lookup in a dict is 1000x faster than numpy.linspace !
        
        :param Dmax: maximum distance/diameter
        :param npt: number of points in the real space (-1)
        
        :return: numpy array of size npt+1 from 0 to Dmax (both included)
        """
        key = RadiusKey(Dmax, npt)
        if key in self.radius_cache:
            value = self.radius_cache[key]
        else:
            value = self.radius_cache[key] = numpy.linspace(0, Dmax, npt+1)
        return value
    
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def calc_trans_matrix(self, double Dmax, int npt):
        """Calculate the matrix T such as:
        
        T.dot.p(r) = I(q)
        
        This is A_ij matrix in equation (2) of Hansen 2000.
        
        This function is memoizing by get_trans_matrix 
        """
        cdef:
            double[:,::1] Tv
            double[::1] q
            double tmp, ql, prefactor, delta_r 
            int l, c, h, w
        q = self.q
        h = self.size
        w = npt + 1
        delta_r = Dmax / npt
        prefactor = 4.0 * pi * delta_r      
        T = numpy.empty((h,w), dtype=numpy.float64)
        Tv = T.view()
        #nogil ?
        for l in range(h):
            ql = q[l] * delta_r
            for c in range(w):
                tmp = ql * c
                tmp = sin(tmp)/tmp if tmp!=0.0 else 1.0
                Tv[l, c] = prefactor * tmp
        return T

    def get_trans_matrix(self,double Dmax, int npt):
        """Get the matrix T form cache or calculates it:
        
        T.dot.p(r) = I(q)
        
        This is A_ij matrix in eq.2 of Hansen 2000.
        
        This function does the memoizing for  calc_trans_matrix
        """
        key = RadiusKey(Dmax, npt)
        if key in self.transfo_cache:
            value = self.transfo_cache[key]
        else:
            value = self.transfo_cache[key] = self.calc_trans_matrix(Dmax, npt)
        return value
    
    def opti_evidence(self, param, npt):
        """Function made for optimization based on the evidence maximisation
        
        :param parm: 2-tuple containing Dmax, log(alpha)
        :param npt: number of points in the real space (-1)
        :return: -evidence for optimisation 
        """ 
        Dmax, logalpha = param
        key = EvidenceKey(Dmax, exp(logalpha), npt)
        if key in self.evidence_cache:
            return -self.evidence_cache[key].evidence
        return -self.calc_evidence(Dmax, exp(logalpha), npt)
    
    def calc_evidence(self,
                      double Dmax, 
                      double alpha, 
                      int npt):
        """
        Calculate the evidence for the given set of parameters
        
        The Evidence is in Bayesian statistics log(P/(1-P)).
        According to eq.17, here evidence is log(P) only
        
        :param Dmax: diameter or longest distance in the object
        :param alpha: smoothing factor (>=0, not its log!)
        :param npt: number of points in the real space (-1)
        :return: evidence (other results are cached) 
        
        All the equation number are refering to 
        J. Appl. Cryst. (2000). 33, 1415-1421 
        """
        cdef:
            double[::1] radius, p_r, f_r, sigma2, sum_dia
            double[:, ::1] B
            double chi2, regularization, xprec, dotsp
        
        xprec = 0.999
        
        #Simple checks: Dmax and alpha need to be positive 
        if Dmax<=0:
            logger.error("Dmax negative: alpha=%s Dmax=%s", alpha, Dmax)
            return -numpy.inf
        if alpha<=0:
            logger.error("alpha negative: alpha=%s Dmax=%s", alpha, Dmax)
            return -numpy.inf
        
        radius = self.radius(Dmax, npt) 
        p_r = self.prior_distribution(self.I0_guess, Dmax, npt) 
        #Note, here p_r is what Hansen calls m in eq.5
        p_r[0] = 0
        f_r = numpy.zeros(npt+1, dtype=numpy.float64)
        #Note: f_r was called P in the original RAW BIFT code
        sigma2 = numpy.zeros(npt+1, dtype=numpy.float64)
                
        transfo_mtx  = self.get_trans_matrix(Dmax, npt)

        #Slightly faster to create this first
        norm_T = transfo_mtx/numpy.asarray(self.variance)[:,None]  
    
        #Creates YSUM in BayesApp code, some kind of calculation intermediate
        #sum_dia = numpy.sum(norm_T*numpy.asarray(self.intensity)[:,None], axis=0)
        # 1D vector of shape (q_size)
        sum_dia = numpy.dot(transfo_mtx.T, numpy.asarray(self.intensity)/numpy.asarray(self.variance))
        sum_dia[0] = 0 
    
        #Creates B(i, j) in BayesApp code
        B = numpy.dot(transfo_mtx.T, norm_T)     
        B[0, :] = 0
        B[:, 0] = 0
        # B is the autocorrelation matrix of the transfo_mtx scaled with the
    
        #Do some kind of rescaling of the input: 
        #This would probably better be done on the large intensity region like Imax> I >Imax/2 
        #c1 = numpy.sum(numpy.sum(transfo_mtx[1:4,1:-1]*p_r[1:-1], axis=1)/self.variance[1:4])
        #c2 = numpy.sum(numpy.asarray(self.intensity[1:4])/numpy.asarray(self.variance[1:4]))
        #print(c2/c1, self.scale_factor(transfo_mtx, p_r, 1, 4))
        self.scale_density(transfo_mtx, p_r, f_r, 1, 4, npt, 1.001)
        
        # Do the optimization
        dotsp = self._bift_inner_loop(f_r, p_r, sigma2, B, alpha, npt, sum_dia, xprec=xprec)

        # Calculate the evidence
        regularization = self.calc_regularization(p_r, f_r, sigma2, npt) # eq19
        #chi2 =numpy.sum((numpy.asarray(self.intensity)[1:-1]-numpy.dot(transfo_mtx[1:-1,1:-1], (f_r)[1:-1]))**2/numpy.asarray(self.variance)[1:-1])/self.size 
        chi2 = self.calc_chi2(transfo_mtx, f_r, npt) #  eq.6 
        rlogdet = self.calc_rlogdet(f_r, B, alpha, npt) # part of eq.20
    
        # The probablility is described in eq. 17, the evidence is apparently log(P) (
        evidence = - log(Dmax) \
                   - alpha*regularization \
                   - 0.5 * chi2 * self.size \
                   - 0.5 * rlogdet \
                   - log(alpha)

        # Some kind of after the fact adjustment
        if evidence <= 0 and dotsp < xprec:
            evidence=evidence*30
        elif dotsp < xprec:
            evidence = evidence/30.

        key = EvidenceKey(Dmax, alpha, npt)
        self.evidence_cache[key] = EvidenceResult(evidence, chi2, regularization, numpy.asarray(radius), numpy.asarray(f_r))
        return evidence
    
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef double  calc_regularization(self,
                                      double[::1] p_r,
                                      double[::1] f_r,
                                      double[::1] sigma2,
                                      int npt):
        """Calculate the regularization factor as defined in eq. 19:
                
        regularization = numpy.sum((f[1:-1]-p[1:-1])**2/sigma2[1:-1])  
        
        :param p_r: smoothed density
        :param f_r: prior density
        :param sigma2: variance of density, i.e. deviation on the density squared 
        
        Nota: the first and last points are skipped as they are null by construction
        """
        cdef:
            int j
            double tmp 
        tmp = 0.0
        for j in range(1, npt):
            tmp += (p_r[j] - f_r[j])**2 / sigma2[j]
        return tmp

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef double calc_chi2(self, 
                           double[:, ::1] transfo,
                           double[::1] density,
                           int npt
                           ):
        """Calculate chi², actually divided by the number of points
        
        This is defined in eq.6
        
        chi² = sum[ (I(q) - Im(q))²/err(q)² ] / len(q)
        
        where Im = T.dot.p(r)
        
        :param transfo: the tranformation matrix T
        :param density: the densty p(r)
        :return: chi²/size
        
        Former implementation:
        chi2 = numpy.sum((i[1:-1]-numpy.sum(T[1:-1,1:-1]*f[1:-1], axis=1))**2/err[1:-1])/i.size
        """                
        cdef:
            int size, idx_q, idx_r
            double chi2, Im
        
        chi2 = 0.0
        size = self.size - 1 
        for idx_q in range(1, size):
            Im = 0.0
            for idx_r in range(1, npt):
                Im += transfo[idx_q, idx_r] * density[idx_r]
            chi2 += ((Im - self.intensity[idx_q])**2/self.variance[idx_q]) 
        return chi2 / self.size 
       
    cpdef double calc_rlogdet(self, 
                             double[::1] f_r,
                             double[:, ::1] B,
                             double alpha,
                             int npt): 
        """
        Calculate the log of the determinant of the the matrix U.
        This is part of the evidence.
        
        u = numpy.sqrt(numpy.abs(numpy.outer(f_r[1:-1], f_r[1:-1])))*B[1:-1, 1:-1]/alpha
        u[numpy.diag_indices(u.shape[0])] += 1
        #w = numpy.linalg.svd(u, compute_uv = False)
        #rlogdet = numpy.sum(numpy.log(numpy.abs(w)))
        rlogdet = log(fabs(numpy.linalg.det(u)))

        :param f_r: density as function of r
        :param B: 
        """
        cdef:
            int j, k
            double[:, ::1] u
        u = numpy.empty((npt-1, npt-1), dtype=numpy.float64)
        for j in range(1, npt):
            for k in range(1, j+1):
                u[j-1, k-1] = u[k-1, j-1] = sqrt(fabs(f_r[j]*f_r[k]))*B[j, k]/alpha + (1.0 if j==k else 0.0)
        #u = numpy.sqrt(numpy.abs(numpy.outer(f_r[1:-1], f_r[1:-1])))*B[1:-1, 1:-1]/alpha
        #u[numpy.diag_indices(u.shape[0])] += 1
        #w = numpy.linalg.svd(u, compute_uv = False)
        return numpy.sum(numpy.log(numpy.abs(numpy.linalg.svd(u, compute_uv = False))))
        #return log(fabs(numpy.linalg.det(u)))  

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)    
    cpdef double scale_density(self, 
                             double[:, ::1] transfo,
                             double[::1] p_r,
                             double[::1] f_r,
                             int start,
                             int stop,
                             int npt,
                             float factor):
        """
        Do some kind of rescaling of the prior density
        
        This allows you to chose which q-regions to scale the data on. 
        The original version says 1-4
        This would probably better be done on the large intensity region like Imax> I >Imax/2
        
        :param tranfo: the matrix which T.p(r) = I(q)
        :param density: the p(r) vector
        :param start: the start point in q-bin
        :param stop: the end point in q-bin
        :return: scale factor
        c1 = numpy.sum(numpy.sum(transfo[1:4,1:-1]*p_r[1:-1], axis=1)/self.variance[1:4])
        c2 = numpy.sum(numpy.asarray(self.intensity[1:4])/numpy.asarray(self.variance[1:4]))
        """
        cdef:
            int j, k
            double num, denom, tmp, v, scale_f, scale_p
        num = denom = 0.0
        for j in range(start, stop):
            v = self.variance[j]
            num += self.intensity[j] / v 
            tmp = 0.0
            for k in range(1, npt):
                tmp += transfo[j, k]*p_r[k]
            
            denom += tmp/v
#         with gil:
#             c1 = numpy.sum(numpy.sum(numpy.dot(transfo[1:4,1:-1],p_r[1:-1])/self.variance[1:4]))
#             c2 = numpy.sum(numpy.asarray(self.intensity[1:4])/numpy.asarray(self.variance[1:4]))            
#             print(c2/c1, num/denom)
        scale_p = num/denom
        scale_f = factor * scale_p
        
        # Update data in place
        for j in range(1, npt):
            v = p_r[j]
            p_r[j] = v * scale_p
            f_r[j] = v * scale_f
        return num/denom
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef inline void smooth_density(self,
                               double[::1] raw,
                               double[::1] smooth,
                               int npt) nogil:
        """This function applies the smoothing of the density plot
        
        :param raw: raw density, called *f* in eq.19 
        :param smooth: smoothed density, called *m* in eq.19
        :param npt: number of points in the real space (-1)
        """
        cdef:
            int k
        for k in range(2, npt-1):
                smooth[k] = 0.5 * (raw[k-1] + raw[k+1])
        smooth[0] = raw[0] = 0.0        # This enforces the boundary values to be null
        smooth[1] = raw[2] * 0.5
        smooth[npt-1] = smooth[npt-2] * 0.5 # is it p or f on the RHS?
        smooth[npt] = raw[npt] = 0.0     # This enforces the boundary values to be null

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef inline double _bift_inner_loop(self,
                                        double[::1] f_r,
                                        double[::1] p_r,
                                        double[::1] sigma2,
                                        double[:, ::1] B,
                                        double alpha,
                                        int npt,
                                        double[::1] sum_dia,
                                        double xprec) nogil:
        """
        Loop and seek for self consistence and smooth f
    
        Inner most function for BIFT ... most time is spent here.
    
        :param f_r: guess density of size N+1, limits are null. Both input and output array
        :param p_r: smoothed version of f over the 2 neighbours. Both input and output array
        :param sigma2: variance (std²) of p_r. Mainly an output array, assuming poissonnian
        :param B: 2D array (N+1 x N+1) with the autocorrelation of the transoformation matrix 
        :param alpha: smoothing factor
        :param npt: number of points in the real space (-1)
        :param maxit: Maximum number of iteration: 2000
        :param minit: Minimum number of iteration: 200
        :param xprec: Precision expected (close to 1, but lower)
        :param omega: Upgrade dumping factor, [0=no update, 1=full update, default 0.5]
        :param epsilon: The minimum density (except at boundaries)
        :return: dotsp
        
        Nota: f_r, p_r and sigma2 are upgraded in place.
        """
        cdef:
            double dotsp, p_k, f_k, tmp_sum, fx, sigma_k, sum_dia_k, B_kk
            double sum_s2, sum_c2, sum_sc, s_k, c_k, sc_k, omega, epsilon
            int j, k, maxit, minit
    
        #Define starting conditions 
        maxit = 2000
        minit = 100
        omega = 0.5
        epsilon = 1e-10
        
        #loop variables
        dotsp = 0.0
        sum_c2 = sum_s2 = sum_sc = 0.0
        #Start loop
        for ite in range(maxit):
            if (ite > minit) and (dotsp > xprec):
                break
    
            #some kind of renormalization of the f & p vector to ensure positivity
            for k in range(1, npt):
                p_k = p_r[k]
                f_k = f_r[k]
                sigma2[k] = fabs(p_k + epsilon)
                if p_k <= 0:
                    p_r[k] = -p_k + epsilon
                if f_k <= 0:
                    f_r[k] = -f_k + epsilon
    
            #Apply smoothness constraint: p is the smoothed version of f
            self.smooth_density(f_r, p_r, npt)
    
            #Calculate the next correction
            for k in range(1, npt):
                # a bunch of local variables:
                f_k = f_r[k]
                p_k = p_r[k]
                sigma_k = sigma2[k]
                sum_dia_k = sum_dia[k]
                B_kk = B[k, k]
    
                # fsumi = numpy.dot(B[k, 1:N], f[1:N]) - B_kk*f_k
                tmp_sum = 0.0
                for j in range(1, npt):
                    tmp_sum += B[k, j] * f_r[j]
                tmp_sum -= B[k, k]*f_r[k]
    
                fx = (2.0*alpha* p_k/sigma_k + sum_dia_k - tmp_sum) / (2.0*alpha/sigma_k + B_kk)
                # Finally update the value
                f_r[k] = f_k = (1.0-omega)*f_k + omega*fx
            #Synchro point
            
            
            # Calculate convergence
            sum_c2 = sum_sc = sum_s2 = 0.0
            for k in range(1, npt):
                s_k = 2.0 * (p_r[k] - f_r[k]) / sigma2[k] #Check homogeneity
                tmp_sum = 0.0
                for j in range(1, npt):
                    tmp_sum += B[k, j] * f_r[j]
                c_k = 2.0 * (tmp_sum - sum_dia[k])
                # There are 3 scalar products:
                sum_c2 += c_k * c_k
                sum_s2 += s_k * s_k
                sum_sc += s_k * c_k
    
            denom = sqrt(sum_s2*sum_c2)
    
    #         gradsi = 2.0*(p[1:N] - f[1:N])/sigma2[1:N]
    #         gradci = 2.0*(numpy.dot(B[1:N,1:N],f[1:N]) - sum_dia[1:N])
    
    #         wgrads2 = numpy.dot(gradsi,gradsi)
    #         wgradc2 = numpy.dot(gradci, gradci)
    #         denom = sqrt(wgrads2*wgradc2)
            if denom == 0.0:
                dotsp = 1.0
            else:
    #             dotsp = numpy.dot(gradsi,gradci) / denom
                dotsp = sum_sc / denom
        return dotsp

    def calc_stats(self):
        """Calculate the statistics on all points encountered so far ....
        
        :return: large namedtuple
        """
        cdef: 
            int samples, npt, best_idx
            double best_evidence, ev_max
            
        samples = len(self.evidence_cache)
        if samples < 2:
            raise RuntimeError("Unable to calculate statistics without evidences having been optimized.")

        best_key, best = self.get_best()
        radius = best.radius
        npt = radius.size
        densities = numpy.zeros((samples, npt), dtype=numpy.float64)
        evidences = numpy.zeros(samples, dtype=numpy.float64)
        Dmaxs = numpy.zeros(samples, dtype=numpy.float64)
        alphas = numpy.zeros(samples, dtype=numpy.float64)
        chi2s = numpy.zeros(samples, dtype=numpy.float64)
        regularizations = numpy.zeros(samples, dtype=numpy.float64)
        
        for idx, key in enumerate(self.evidence_cache):
            Dmaxs[idx] = key.Dmax
            alphas[idx] = key.alpha
            value = self.evidence_cache[key]
            evidences[idx] = value.evidence
            chi2s[idx] =  value.chi2
            regularizations[idx] = value.regularization
            densities[idx] = numpy.interp(radius, value.radius, value.density, 0,0)

        #Then, calculate the probability of each result as exp(evidence - evidence_max)**(1/minimum_chisq), normalized by the sum of all result probabilities
        ev_max = evidences.max()
        proba = numpy.exp(evidences - ev_max)**(1./chi2s.min())
        proba /= proba.sum()
            
        #Then, calculate the average P(r) function as the weighted sum of the P(r) functions
        density_avg = numpy.sum(densities*proba[:,None], axis=0)
        #Then, calculate the error in P(r) as the square root of the weighted sum of squares of the difference between the average result and the individual estimate
        density_std = numpy.sqrt(numpy.abs(numpy.sum((densities-density_avg)**2*proba[:,None], axis=0)))
    
        #Then, calculate structural results as weighted sum of each result
        evidence_avg = numpy.dot(evidences, proba)
        evidence_std = numpy.sqrt(numpy.dot((evidences - evidence_avg)**2, proba))

        Dmax_avg = numpy.dot(Dmaxs, proba)
        Dmax_std = numpy.sqrt(numpy.dot((Dmaxs - Dmax_avg)**2, proba))

        alpha_avg = numpy.dot(alphas, proba)
        alpha_std = numpy.sqrt(numpy.dot((alphas - alpha_avg)**2, proba))
        
        chi2_avg = numpy.dot(chi2s, proba)
        chi2_std = numpy.sqrt(numpy.dot((chi2s - chi2_avg)**2, proba))
        
        regularization_avg = numpy.dot(regularizations, proba)
        regularization_std = numpy.sqrt(numpy.dot((regularizations - regularization_avg)**2, proba))
        
        areas = numpy.trapz(densities, radius, axis=1)
        area2s = numpy.trapz(densities*radius**2, radius, axis=1)
        
        Rgs = numpy.sqrt(area2s/(2.*areas))
        Rg_avg = numpy.sum(Rgs*proba)
        Rg_std = numpy.sqrt(numpy.sum((Rgs - Rg_avg)**2*proba))
        
        area = numpy.sum(areas*proba)
        I0_avg = 4.0 * pi * area 
        I0_std = 4.0 * pi * numpy.sqrt(numpy.sum((areas - area)**2*proba))
    
        #Should I also extrapolate to q=0? Might be good, though maybe not in this function
        #Should I report number of good parameters (ftot(nmax-12 in Hansen code, line 2247))
        #Should I report number of Shannon Channels? That's easy to calculate: q_range*dmax/pi
        return StatsResult(radius, density_avg, density_std, 
                           evidence_avg, evidence_std, 
                           Dmax_avg, Dmax_std,
                           alpha_avg, alpha_std,
                           chi2_avg, chi2_std,
                           regularization_avg, regularization_std,
                           Rg_avg, Rg_std,
                           I0_avg, I0_std)
    
    