# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
#This is for developping
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
"""
Bayesian Inverse Fourier Transform

This code is the implementation of
Steen Hansen J. Appl. Cryst. (2000). 33, 1415-1421

Based on the BIFT from Jesse Hopkins, available at:
https://sourceforge.net/p/bioxtasraw/git/ci/master/tree/bioxtasraw/BIFT.py

This is a major rewrite in Cython
"""
cdef:
    list authors
    str __license__, __copyright__, __date__

__authors__ = ["Jerome Kieffer", "Jesse Hopkins"]
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "10/06/2020"

import time
import cython
from cython.parallel import prange
from cython.view cimport array as cvarray
import numpy
cimport numpy as cnumpy
from libc.math cimport sqrt, fabs, pi, sin, log, exp, isfinite

from scipy.linalg import lapack
from scipy.linalg.cython_lapack cimport dgesvd
from scipy.linalg.cython_blas cimport dgemm, ddot
import logging
import itertools
logger = logging.getLogger(__name__)

from .collections import RadiusKey, PriorKey, TransfoValue, EvidenceKey, EvidenceResult, StatsResult

################################################################################
# BLAS / LAPACK wrappers
################################################################################

cpdef inline double blas_ddot(double[::1] a, double[::1] b) nogil:
    "Wrapper for double precision dot product"
    cdef:
        int n, one=1
        double *a0=&a[0]
        double *b0=&b[0]

    n = a.shape[0]
    if n != b.shape[0]:
        with gil:
            raise ValueError("Shape mismatch in input arrays.")

    return ddot(&n, a0, &one, b0, &one)


cpdef int blas_dgemm(double[:,::1] a, double[:,::1] b, double[:,::1] c, double alpha=1.0, double beta=0.0) nogil except -1:
    "Wrapper for double matrix-matrix multiplication C = AxB "
    cdef:
        char *transa = 'n'
        char *transb = 'n'
        int m, n, k, lda, ldb, ldc
        double *a0=&a[0,0]
        double *b0=&b[0,0]
        double *c0=&c[0,0]

    ldb = (&a[1,0]) - a0 if a.shape[0] > 1 else 1
    lda = (&b[1,0]) - b0 if b.shape[0] > 1 else 1

    k = b.shape[0]
    if k != a.shape[1]:
        with gil:
            raise ValueError("Shape mismatch in input arrays.")
    m = b.shape[1]
    n = a.shape[0]
    if n != c.shape[0] or m != c.shape[1]:
        with gil:
            raise ValueError("Output array does not have the correct shape.")
    ldc = (&c[1,0]) - c0 if c.shape[0] > 1 else 1
    dgemm(transa, transb, &m, &n, &k, &alpha, b0, &lda, a0,
               &ldb, &beta, c0, &ldc)
    return 0


cpdef int lapack_svd(double[:, ::1] A, double[::1] eigen, double[::1] work) nogil except -1:
    cdef:
        char *jobN = 'n'
        int n, lda, lwork, info, one=1

    info = 0
    n = A.shape[0]
    lwork = work.shape[0]
    if n != A.shape[1]:
        return -1
    lda = (&A[1,0]) - &A[0,0] if A.shape[0] > 1 else 1
    dgesvd(jobN, jobN, &n, &n, &A[0,0], &lda, &eigen[0], &work[0] , &one, &work[0], &one, &work[0], &lwork, &info)
    if info:
        return -1
    return 0

################################################################################
# Helper functions
################################################################################

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
    p_r = cvarray(shape=(npt+1,), itemsize=sizeof(double), format="d")
    with nogil:
        norm = I0 / (4.0 * pi * Dmax**3 / 24.0)
        delta_r = Dmax / npt
        for j in range(npt+1):
            r = j * delta_r
            p_r[j] = norm * r**2 * (1.0 - 1.5*(r/Dmax) + 0.5*(r/Dmax)**3)
    # p = p * I0/(4*pi*Dmax**3/24.)
    # p = p * I0/(Dmax**3/(24.*(r[1]-r[0])))   #Which normalization should I use? I'm not sure either agrees with what Hansen does.
    return numpy.asarray(p_r)


cpdef inline distribution_parabola(double I0,
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
    p_r = cvarray(shape=(npt+1,), itemsize=sizeof(double), format="d")

    with nogil:
        norm = I0 / (4.0 * pi * Dmax**3 / 6.0)
        delta_r = Dmax / npt
        for j in range(npt+1):
            r = j * delta_r
            p_r[j] = norm * r * (Dmax - r)
    return numpy.asarray(p_r)


cpdef inline double  calc_regularization(double[::1] p_r,
                                         double[::1] f_r,
                                         double[::1] sigma2) nogil:
    """Calculate the regularization factor as defined in eq. 19:

    regularization = numpy.sum((f[1:-1]-p[1:-1])**2/sigma2[1:-1])

    :param p_r: smoothed density
    :param f_r: prior density
    :param sigma2: variance of density, i.e. deviation on the density squared

    Nota: the first and last points are skipped as they are null by construction
    """
    cdef:
        int j, npt
        double tmp
    tmp = 0.0
    npt = p_r.shape[0] - 1
    for j in range(1, npt):
        tmp += (p_r[j] - f_r[j])**2 / sigma2[j]
    return tmp

cpdef inline double calc_rlogdet(double[::1] f_r,
                                 double[:, ::1] B,
                                 double alpha,
                                 double[:, ::1] U,
                                 double[::1] eigen,
                                 double[::1] work) nogil:
        """
        Calculate the log of the determinant of the the matrix U.
        This is part of the evidence.

        u = numpy.sqrt(numpy.abs(numpy.outer(f_r[1:-1], f_r[1:-1])))*B[1:-1, 1:-1]/alpha
        u[numpy.diag_indices(u.shape[0])] += 1
        w = numpy.linalg.svd(u, compute_uv = False)
        rlogdet = numpy.sum(numpy.log(numpy.abs(w)))

        :param f_r: density as function of r
        :param B: autocorrelation of the transformation matrix
        :param alpha: weight of the regularization
        :param npt: number of points (-1) for the density
        :param U: squarre matrix (npt-1, npt-1) for calculating the determinant
        :param eigen: vector with the eigenvalues of matrix U (size npt-1)
        :param work: some work-space buffer used by LAPACK for the SVD
        :return:  log(abs(det(U))) where
        """
        cdef:
            int j, k, npt
            double rlogdet
        npt = f_r.shape[0]-1
        for j in range(1, npt):
            for k in range(1, npt):
                U[j-1, k-1] = sqrt(fabs(f_r[j]*f_r[k]))*B[j, k]/alpha + (1.0 if j==k else 0.0)
        if lapack_svd(U, eigen, work):
            with gil:
                raise RuntimeError("SVD failed")
        rlogdet = 0.0
        for j in range(npt-1):
            rlogdet += log(fabs(eigen[j]))
        return rlogdet

cpdef inline void ensure_edges_zero(double[::1] distribution) nogil:
  """This function sets the first and last point of the density plot to 0
  :param distribution: raw density
  The operation is performed in place
  """
  npt = distribution.shape[0] - 1
  distribution[0] = 0.0
  distribution[npt] = 0.0

cpdef inline void smooth_density(double[::1] raw,
                                 double[::1] smooth) nogil:
        """This function applies the smoothing of the density plot

        :param raw: raw density, called *f* in eq.19
        :param smooth: smoothed density, called *m* in eq.19

        The smoothing is performed in place
        """
        cdef:
            int k, npt
        #assert raw.shape[0] == smooth.shape[0]
        npt = raw.shape[0] - 1
        #Set the edges to be equal
        smooth[0] = raw[0]
        smooth[npt] = raw[npt]
        for k in range(2, npt-1):
            smooth[k] = 0.5 * (raw[k-1] + raw[k+1])
        #Interpolate the second and second to last point
        smooth[1] = (smooth[0] + smooth[2]) * 0.5
        smooth[npt-1] = (smooth[npt-2] + smooth[npt]) * 0.5

################################################################################
# Main class
################################################################################

cdef class BIFT:
    """Bayesian Inverse Fourier Transform

    :param q: scattering vector in 1/nm or 1A, the unit of q imposes the one on Dmax, r, ...
    :param I: Scattering intensity I(q)
    :param I_std: error on the intensity estimation    """
    cdef:
        readonly int size, high_start, high_stop
        readonly double I0_guess, delta_q, Dmax_guess, alpha_max
        readonly double[::1] q, intensity, variance, wisdom
        readonly dict prior_cache, evidence_cache, radius_cache, transfo_cache, lapack_cache

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
        self.wisdom = None
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
        self.lapack_cache = {}


    def __dealloc__(self):
        "This is the destructor of the class: free the memory and empty all caches"
        self.reset()
        self.q = self.intensity = self.variance = None

    def reset(self):
        "rest all caches"
        cdef:
            dict cache
        for cache  in (self.prior_cache,  self.evidence_cache, self.radius_cache, self.transfo_cache, self.lapack_cache):
            if cache is not None:
                for key in list(cache.keys()):
                    cache.pop(key)

    def set_Guinier(self, guinier_fit, Dmax_over_Rg=3.0):
        """Set some starting point from Guinier fit like:

        Dmax = 3 Rg
        I0 for prior density modeling
        Guinier region for high signal with low noise section

        :param guinier_fit: RG_RESULT instance from autorg fit
        :param Dmax_over_Rg: guess the Dmax =  factor * Rg
        :return: guessed Dmax
        """
        self.I0_guess = guinier_fit.I0
        self.high_start = guinier_fit.start_point
        self.high_stop = guinier_fit.end_point
        self.Dmax_guess = guinier_fit.Rg * Dmax_over_Rg
        return self.Dmax_guess


    def guess_alpha_max(self, int npt):
        """This is to define the maximum realistic alpha value to scan for.

        idea: limit case is alpha·S0 = chi2/2

        so calculate the prior p, the regularization factor associated, ...

        :param npt: number of points in the real space (-1)
        """
        cdef:
            double[::1] density, smooth
            double[:, ::1] transfo
            double regularization, chi2
        if self.Dmax_guess<=0.0:
            raise RuntimeError("Please initialize with Guinier fit data using set_Guinier")
        density = self.prior_distribution(self.I0_guess, self.Dmax_guess, npt)
        ensure_edges_zero(density)
        smooth = numpy.zeros(npt+1, numpy.float64)
        smooth_density(density, smooth)
        regularization = calc_regularization(density, smooth, density) # eq19
        transfo = self.get_transformation_matrix(self.Dmax_guess, npt)
        chi2 = self.calc_chi2(transfo, density, npt)
        return 0.5*chi2/regularization

    def get_best(self):
        """Return the most probable configuration found so far and the number of valid
        """
        cdef:
            double best_evidence
            int nvalid
        best_evidence = numpy.finfo(numpy.float64).min
        best_key = None
        nvalid = 0
        for key, value in self.evidence_cache.items():
            if value.converged:
                nvalid += 1
                if value.evidence>best_evidence:
                    best_key = key
                    best_evidence = value.evidence
        if nvalid == 0:
            "None have converged yet ... take the least worse answer"
            for key, value in self.evidence_cache.items():
                if value.evidence>best_evidence:
                    best_key = key
                    best_evidence = value.evidence

        return best_key, self.evidence_cache.get(best_key), nvalid

    def prior_distribution(self,
                           double I0,
                           double Dmax,
                           int npt,
                           dist_type='sphere'):
        """Calculate the prior distribution for the bayesian analysis

        Implements memoizing.
        The memoising is performed with I0 = Dmax = 1. Rescaling is performd a posterori

        :param I0: forward scattering intensity, often approximated by th maximum intensity
        :param Dmax: Largest dimention of the object
        :param npt: number of points in the real space (-1)
        :param dist_type: Implements "sphere" and "parabola". "wisdom" is possible after initialization
        :return: the distance distribution function p(r) where r = numpy.linspace(0, Dmax, npt+1)

        Nota: the wisdom is the normalized best density found. It needs to be manually updated with update_wisdom()
        """
        key = PriorKey(dist_type, npt)
        if key in self.prior_cache:
            value = self.prior_cache[key]
        else:
            if dist_type == "sphere":
                value = self.prior_cache[key] = distribution_sphere(1, 1, npt)
            elif dist_type == "parabola":
                value = self.prior_cache[key] = distribution_parabola(1, 1, npt)
            else:
                raise RuntimeError("Only 'sphere' is accepted for dist_type")
        return (I0/Dmax) * value

    def update_wisdom(self):
        best_key, best_value, nvalid  =self.get_best()
        npt = best_key.npt
        if nvalid == 0:
            logger.warning("No converged solution was found. It is not advices to ")
            density = distribution_sphere(1, 1, npt)
        else:
            density = best_value.density / (4.0*pi*numpy.trapz(best_value.density, numpy.linspace(0, 1, npt+1)))
        key = PriorKey("wisdom", npt)
        self.prior_cache[key] = density

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

    def get_transformation_matrix(self,
                                  double Dmax,
                                  int npt,
                                  bint all_=False):
        """Get the matrix T from the cache or calculates it:

        T.dot.p(r) = I(q)

        This is A_ij matrix in eq.2 of Hansen 2000.

        This function does the memoizing for T, B = T.T x T) and sum_dia

        :param Dmax: diameter or longest distance in the object
        :param npt: number of points in the real space (-1)
        :param all_: return in addition B and sum_dia arrays
        :return: the T matrix as: T.dot.p(r) = I(q)
        """
        cdef:
            double[::1] sum_dia
            double[:, ::1] B, transpo_mtx, transfo_mtx
        key = RadiusKey(Dmax, npt)
        if key in self.transfo_cache:
            value = self.transfo_cache[key]
        else:
            transfo_mtx  = cvarray(shape=(self.size, npt+1), itemsize=sizeof(double), format="d")
            transpo_mtx  = cvarray(shape=(npt+1, self.size), itemsize=sizeof(double), format="d")
            B = cvarray(shape=(npt+1, npt+1), itemsize=sizeof(double), format="d")
            sum_dia = cvarray(shape=(npt+1,), itemsize=sizeof(double), format="d")
            with nogil:
                self.initialize_arrays(Dmax, npt, transfo_mtx, transpo_mtx, B, sum_dia)
            value = self.transfo_cache[key] = TransfoValue(numpy.asarray(transfo_mtx), numpy.asarray(B), numpy.asarray(sum_dia))
        if all_:
            return value
        else:
            return value.transfo

    def get_workspace_size(self, npt):
        "This function calls LAPACK to measure the size of the workspace needed for the SVD"
        if npt in self.lapack_cache:
            value = self.lapack_cache[npt]
        else:
            _, s_lw = lapack.get_lapack_funcs(('gesvd', 'gesvd_lwork'))
            self.lapack_cache[npt] = value = lapack._compute_lwork(s_lw, npt, npt)
        return value

    cdef int initialize_arrays(self,
                            double Dmax,
                            int npt,
                            double[:, ::1] transf_matrix,
                            double[:, ::1] transp_matrix,
                            double[:, ::1] B,
                            double[::1] sum_dia
                            ) nogil except -1:

        cdef:
            double tmp, ql, prefactor, delta_r, il, varl
            int l, c, res

        delta_r = Dmax / npt
        prefactor = 4.0 * pi * delta_r
        sum_dia[:] = 0.0
        for l in range(self.size):
            ql = self.q[l] * delta_r
            il = self.intensity[l]
            varl = self.variance[l]
            for c in range(npt+1):
                tmp = ql * c
                tmp = prefactor * (sin(tmp)/tmp if tmp!=0.0 else 1.0)
                transf_matrix[l, c] = tmp
                sum_dia[c] += tmp * il / varl
                transp_matrix[c, l] = tmp / varl
        sum_dia[0] = 0.0

        res = blas_dgemm(transp_matrix, transf_matrix, B)
        if res:
            return -1
        #B = numpy.dot(TnT, T)
        B[0, :] = 0.0
        B[:, 0] = 0.0
        return 0

    def opti_evidence(self, param,
                      int npt, bint prior=0):
        """Function made for optimization based on the evidence maximisation

        :param parm: 2-tuple containing Dmax, log(alpha)
        :param npt: number of points in the real space (-1)
        :param prior: By default (False) use the sphere density to start with
                      Set to True to use the shape of the best found structure
        :return: -evidence for optimisation
        """
        cdef:
            double Dmax, logalpha
        Dmax, logalpha = param
        alpha = exp(logalpha)
        key = EvidenceKey(Dmax, alpha, npt)
        if key in self.evidence_cache:
            return -self.evidence_cache[key].evidence
        return -self.calc_evidence(Dmax, alpha, npt)

    cpdef double calc_evidence(self,
                               double Dmax,
                               double alpha,
                               int npt,
                               bint prior=False) with gil:
        """
        Calculate the evidence for the given set of parameters

        The Evidence is in Bayesian statistics log(P/(1-P)).
        According to eq.17, here evidence is log(P) only

        :param Dmax: diameter or longest distance in the object
        :param alpha: smoothing factor (>=0, not its log!)
        :param npt: number of points in the real space (-1)
        :param prior: By default (False) use the sphere density to start with
                      Set to True to use the shape of the best found structure
        :return: evidence (other results are cached)

        All the equation number are refering to
        J. Appl. Cryst. (2000). 33, 1415-1421
        """
        cdef:
            double[::1] radius, p_r, f_r, sigma2, sum_dia, workspace, eigen
            double[:, ::1] B, transfo_mtx, U
            double chi2, regularization, xprec, dotsp, rlogdet, evidence
            int j
            bint is_valid, converged

        xprec = 0.999
        key = EvidenceKey(Dmax, alpha, npt)
        #Simple checks: Dmax and alpha need to be positive
        if Dmax<=0:
            logger.info("Dmax negative: alpha=%s Dmax=%s", alpha, Dmax)
            self.evidence_cache[key] = EvidenceResult(-numpy.inf, numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN, False)
            return -numpy.inf
        if alpha<=0:
            logger.info("alpha negative: alpha=%s Dmax=%s", alpha, Dmax)
            self.evidence_cache[key] = EvidenceResult(-numpy.inf, numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN, False)
            return -numpy.inf

        # Here we perform all memory allocation for the complete function

        radius = self.radius(Dmax, npt)
        p_r = self.prior_distribution(self.I0_guess, Dmax, npt, dist_type="wisdom" if prior else "sphere")

        #here p_r is what Hansen calls m in eq.5
        p_r[0] = 0
        f_r = cvarray(shape=(npt+1,), itemsize=sizeof(double), format="d")
        #Note: f_r was called P in the original RAW BIFT code
        sigma2 = cvarray(shape=(npt+1,), itemsize=sizeof(double), format="d")

        #Those are used by the rlogdet function to calculate the determinant
        U = cvarray(shape=(npt-1, npt-1), itemsize=sizeof(double), format="d")
        eigen = cvarray(shape=(npt-1,), itemsize=sizeof(double), format="d")
        workspace = cvarray(shape=(self.get_workspace_size(npt-1),), itemsize=sizeof(double), format="d")

        transfo_mtx, B, sum_dia = self.get_transformation_matrix(Dmax, npt, all_=True)

        # At this stage, all buffers have been allocated ...
        with nogil:
            #Do some kind of rescaling of the input:
            #This would probably better be done on the large intensity region like Imax> I >Imax/2
            #c1 = numpy.sum(numpy.sum(transfo_mtx[1:4,1:-1]*p_r[1:-1], axis=1)/self.variance[1:4])
            #c2 = numpy.sum(numpy.asarray(self.intensity[1:4])/numpy.asarray(self.variance[1:4]))
            #print(c2/c1, self.scale_factor(transfo_mtx, p_r, 1, 4))
            self.scale_density(transfo_mtx, p_r, f_r, self.high_start, self.high_stop, npt, 1.001)

            # Do the optimization
            dotsp = self._bift_inner_loop(f_r, p_r, sigma2, B, alpha, npt, sum_dia, xprec=xprec)

            regularization = calc_regularization(p_r, f_r, sigma2) # eq19
            #chi2 =numpy.sum((numpy.asarray(self.intensity)[1:-1]-numpy.dot(transfo_mtx[1:-1,1:-1], (f_r)[1:-1]))**2/numpy.asarray(self.variance)[1:-1])/self.size
            chi2 = self.calc_chi2(transfo_mtx, f_r, npt) #  eq.6
            rlogdet = calc_rlogdet(f_r, B, alpha, U, eigen, workspace) # part of eq.20

            # The probablility is described in eq. 17, the evidence is apparently log(P)
            evidence = - log(Dmax) \
                       - alpha*regularization \
                       - 0.5 * chi2 \
                       - 0.5 * rlogdet \
                       - log(alpha)

            # Some kind of after the fact adjustment
            converged = (dotsp > xprec) # handles the case dotsp is Nan
            if not converged:
                # Make result much less likely, why 30 ?
                if evidence <= 0:
                    evidence *= 30.0
                else:
                    evidence /= 30.0

            #Check if those data are valid
            is_valid = isfinite(evidence)
            for j in range(npt+1):
                is_valid &= isfinite(f_r[j])
        # Store the results into the cache with the GIL
        if is_valid:
            self.evidence_cache[key] = EvidenceResult(evidence,
                                                      chi2/(self.size - npt),
                                                      regularization,
                                                      numpy.asarray(radius),
                                                      numpy.asarray(f_r),
                                                      converged)
            return evidence
        else:
            logger.info("Invalid evidence: Dmax: %s alpha: %s S: %s chi2: %s rlogdet:%s", Dmax, alpha, regularization, chi2, rlogdet)
            self.evidence_cache[key] = EvidenceResult(-numpy.inf, numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN, False)
            return -numpy.inf

    cdef double calc_chi2(self,
                           double[:, ::1] transfo,
                           double[::1] density,
                           int npt
                           )nogil:
        """Calculate chi²

        This is defined in eq.6

        chi² = sum[ (I(q) - Im(q))²/err(q)² ]

        where Im = T.dot.p(r)

        :param transfo: the tranformation matrix T
        :param density: the densty p(r)
        :return: chi²/size

        Former implementation:
        chi2 = numpy.sum((i[1:-1]-numpy.sum(T[1:-1,1:-1]*f[1:-1], axis=1))**2/err[1:-1])/i.size
        used to return the reduced chi², now the not reduced one
        """
        cdef:
            int idx_q, idx_r
            double chi2, Im

        chi2 = 0.0
        for idx_q in range(self.high_start, self.size):
            # Replace with dot-product
            Im = 0.0
            for idx_r in range(1, npt):
                Im += transfo[idx_q, idx_r] * density[idx_r]
            chi2 += ((Im - self.intensity[idx_q])**2/self.variance[idx_q])
        return chi2

    cdef double scale_density(self,
                             double[:, ::1] transfo,
                             double[::1] p_r,
                             double[::1] f_r,
                             int start,
                             int stop,
                             int npt,
                             float factor) nogil:
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
            int j
            double num, denom, tmp, v, scale_f, scale_p
        num = denom = 0.0
        for j in range(start, stop):
            v = self.variance[j]
            num += self.intensity[j] / v
            # Use a dot product
            tmp = blas_ddot(transfo[j, 1:npt], p_r[1:npt])
#             tmp = 0.0
#             for k in range(1, npt):
#                 tmp += transfo[j, k]*p_r[k]

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
            double sum_s2, sum_c2, sum_sc, s_k, c_k, omega, epsilon, denom
            int ite, k, maxit, minit
            bint is_valid

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
            is_valid = True
            #some kind of renormalization of the f & p vector to ensure positivity
            for k in range(1, npt):
                p_k = p_r[k]
                f_k = f_r[k]
                sigma2[k] = fabs(p_k + epsilon)
                if p_k <= 0:
                    p_r[k] = -p_k + epsilon
                if f_k <= 0:
                    f_r[k] = -f_k + epsilon

            #Set edges of f to zero
            ensure_edges_zero(f_r)
            #Apply smoothness constraint: p is the smoothed version of f
            smooth_density(f_r, p_r)

            #Calculate the next correction
            for k in range(1, npt):
                # a bunch of local variables:
                f_k = f_r[k]
                p_k = p_r[k]
                sigma_k = sigma2[k]
                sum_dia_k = sum_dia[k]
                B_kk = B[k, k]

                # fsumi = numpy.dot(B[k, 1:N], f[1:N]) - B_kk*f_k
#                 tmp_sum = 0.0
#                 for j in range(1, npt):
#                     tmp_sum += B[k, j] * f_r[j]
                tmp_sum = blas_ddot(B[k, 1:npt], f_r[1:npt])
                tmp_sum -= B[k, k]*f_r[k]

                fx = (2.0*alpha*p_k/sigma_k + sum_dia_k - tmp_sum) / (2.0*alpha/sigma_k + B_kk)
                is_valid &= isfinite(fx)
                # Finally update the value
                f_r[k] = f_k = (1.0-omega)*f_k + omega*fx
            #Synchro point

#             if not is_valid:
#                 with gil:
#                     for i in range(npt+1):
#                         print(i, f_r[i], p_r[i], sigma2[i])
#                 return 0.0

            # Calculate convergence
            sum_c2 = sum_sc = sum_s2 = 0.0
            for k in range(1, npt):
                s_k = 2.0 * (p_r[k] - f_r[k]) / sigma2[k] #Check homogeneity

#                 tmp_sum = 0.0
#                 for j in range(1, npt):
#                     tmp_sum += B[k, j] * f_r[j]
                tmp_sum = blas_ddot(B[k, 1:npt], f_r[1:npt])

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

    def grid_scan(self,
                  double dmax_min, double dmax_max, int dmax_cnt,
                  double alpha_min, double alpha_max, int alpha_cnt,
                  int npt):
        """Perform a quick scan for Dmax and alpha

        :param alpha_min, alpha_max, alpha_cnt: parameters for a numpy.geomspace scan in alpha
        :param dmax_min, dmax_max, dmax_cnt: parameters for a numpy.linspace scan in Dmax
        :param npt: the number of points in the
        :return: best combination
        """
        cdef:
            double[:, ::1] grid
            double[::1] results, dmax_array, alpha_array
            int idx, best, steps = dmax_cnt*alpha_cnt,
            double Dmax, alpha
        dmax_array = numpy.linspace(dmax_min, dmax_max, dmax_cnt)
        alpha_array = numpy.geomspace(alpha_min, alpha_max, alpha_cnt)
        grid = numpy.array(list(itertools.product(dmax_array, alpha_array)))

        results = numpy.zeros(steps, dtype=numpy.float64)
#         for idx in range(steps):
        with nogil:
            for idx in prange(steps):
                Dmax = grid[idx, 0]
                alpha = grid[idx, 1]
                results[idx] += self.calc_evidence(Dmax, alpha, npt)
        best = numpy.argmax(results)
#         for i, j in zip(grid, results):
#             print(j, i[0], i[1], )
        return EvidenceKey(grid[best, 0], grid[best, 1], npt)

    def monte_carlo_sampling(self,
                             int samples,
                             double nsigma,
                             int npt
                             ):
        """Perform a monte-carlo sampling for Dmax and alpha around the optimum to be able to calculate the statistic of observables

        Dmax is sampled linearly
        alpha is sampled in log-scale

        Re-uses the results from the steepest descent to guess the std of alpha and Dmax.

        Re use the best density as prior for the density

        :param npt: the number of points in the modelisation
        :param samples: number of samples to be taken
        :param nsigma: sample alpha and Dmax at avg ± nx sigma
        :return: Statistics calculated over all explored space
        """
        cdef:
            double[::1] Dmax_samples, alpha_samples
            double[::1] results
            int idx
            double Dmax, alpha, t0, eps
        stats = self.calc_stats()
        if samples == 0:
            return stats
        self.update_wisdom()
        if stats.Dmax_std>0 and stats.Dmax_avg/stats.Dmax_std < nsigma:
            nsigma = stats.Dmax_avg/stats.Dmax_std
            logger.info("Clipping to nsigma=%.2f due to large noise on Dmax: avg=%.2f, std=%.2f", nsigma, stats.Dmax_avg, stats.Dmax_std)
        log_alpha = log(stats.alpha_avg)
        dlog_alpha = stats.alpha_std/stats.alpha_avg
        Dmax_samples = stats.Dmax_avg + nsigma*(2.0*numpy.random.random(samples)-1.0)*stats.Dmax_std
        alpha_samples = numpy.exp(log_alpha + nsigma*(2.0*numpy.random.random(samples)-1.0)*dlog_alpha)
        results = numpy.zeros(samples, dtype=numpy.float64)
        t0 = time.perf_counter()
        with nogil:
            for idx in prange(samples):
                Dmax = Dmax_samples[idx]
                alpha = alpha_samples[idx]
                results[idx] = self.calc_evidence(Dmax, alpha, npt, prior=1)
        logger.debug("Monte-carlo: %i samples at %.2fms/sample", samples, (time.perf_counter()-t0)*1000.0/samples)
        return self.calc_stats()

    def calc_stats(self):
        """Calculate the statistics on all points encountered so far ....

        :return: large namedtuple
        """
        cdef:
            int npt, nvalid, idx
            double area, ev_max, evidence_avg, evidence_std,
            double Dmax_avg, Dmax_std, alpha_avg, alpha_std, chi2_avg, chi2_std,
            double regularization_avg, regularization_std, Rg_std, Rg_avg, I0_avg, I0_std
            cnumpy.ndarray radius, densities, evidences, Dmaxs, alphas, chi2s, regularizations, proba, density_avg, density_std, areas, area2s, Rgs

        best_key, best, nvalid = self.get_best()
        if nvalid < 2:
            raise RuntimeError("Unable to calculate statistics without evidences having been optimized.")

        radius = best.radius
        npt = radius.size
        densities = numpy.zeros((nvalid, npt), dtype=numpy.float64)
        evidences = numpy.zeros(nvalid, dtype=numpy.float64)
        Dmaxs = numpy.zeros(nvalid, dtype=numpy.float64)
        alphas = numpy.zeros(nvalid, dtype=numpy.float64)
        chi2s = numpy.zeros(nvalid, dtype=numpy.float64)
        regularizations = numpy.zeros(nvalid, dtype=numpy.float64)

        idx = 0
        for key in self.evidence_cache:
            value = self.evidence_cache[key]
            if value.converged:
                Dmaxs[idx] = key.Dmax
                alphas[idx] = key.alpha
                evidences[idx] = value.evidence
                chi2s[idx] =  value.chi2r
                regularizations[idx] = value.regularization
                densities[idx] = numpy.interp(radius, value.radius, value.density, 0,0)
                idx+=1

        # Then, calculate the probability of each result as exp(evidence - evidence_max)**(1/minimum_chisq),
        # normalized by the sum of all result probabilities
        proba = numpy.exp(evidences - best.evidence)**(1./chi2s.min())
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
