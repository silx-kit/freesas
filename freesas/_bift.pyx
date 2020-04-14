# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, cdivision=True, embedsignature=True, language_level=3

__author__ = "Jerome Kieffer"
__license__ = "GPL"
__copyright__ = "2020, ESRF"
__date__ = "14/04/2020"


import cython
import numpy
from libc.math cimport sqrt, fabs, pi

# cdef class BIFT:
#     cdef:
#         public int size
#         public double Imax
#         public double[::1] q, intensity, variance
#     
#     def __cinit__(self, q, I, I_std):
#         """Constructor of the Cython class
#         :param q: scattering vector 
#         TODO
#         """
#         cdef:
#             int j
#         self.size = q.size
#         assert self.size == I.size, "Intensity array matches in size"
#         assert self.size == I_std.size, "Error array matches in size"
#         self.q = numpy.ascontiguousarray(q, dtype=numpy.float64)
#         self.intensity = numpy.ascontiguousarray(I, dtype=numpy.float64)
#         self.variance = numpy.ascontiguousarray(I_std**2, dtype=numpy.float64)
#         self.Imax = numpy.max(I)
#         
#     def __dealloc__(self):
#         self.q = self.intensity = self.variance = None

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef distribution_sphere(double I0, 
                          int npt, 
                          double Dmax):
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
    :param npt: Number of points for the distribution
    :param Dmax: Diameter of the the object, here a sphere
    :return: the density p(r)
    """
    cdef:
        int j
        double norm, delta_r, r 
        double[::1] p_r
        
    norm = I0 / (4.0 * pi / 24. * Dmax**3)
    delta_r = Dmax/npt

    p_r = numpy.empty(npt+1, dtype=numpy.float64)
    
    for j in range(npt+1):
        r = j * delta_r
        p_r[j] = norm * r**2 * (1.0 - 1.5*(r/Dmax) + 0.5*(r/Dmax)**3)
    # p = p * I0/(4*pi*Dmax**3/24.)
    # p = p * I0/(Dmax**3/(24.*(r[1]-r[0])))   #Which normalization should I use? I'm not sure either agrees with what Hansen does.

    return numpy.asarray(p_r)


def prior_distribution(double I0, 
                       int npt, 
                       double Dmax, 
                       dist_type='sphere'):
    """Calculate the prior distribution for the bayesian analysis
    
    :param I0: forward scattering intensity, often approximated by th maximum intensity
    :param npt: number of points in the distribution
    :param Dmax: Largest dimention of the object
    :param dist_type: str, for now only "sphere" is acceptable
    :return: the density p(r) where r = numpy.linspace(0, Dmax, 
    """
    if dist_type == 'sphere':
        return distribution_sphere(I0, npt, Dmax)
    raise RuntimeError("Only 'sphere' is accepted for dist_type")


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


def calc_evidence(double Dmax, 
                  double alpha, 
                  double[::1] q, 
                  double[::1] I, 
                  double[::1] I_std, 
                  int npt):
    """This is core function of BIFT which calculate the 
    
    :param Dmax: Diameter of the macromolecule, in nm if q is in inverse nm.
    :param alpha: Smoothness parameter of the curve (no log here !)
    :param q: scattering vector in nm^-1 or A^-1
    :param I: Scattering intensity. Does the unit matter?
    :param I_std: Errors on the measured intensity
    :param npt: number of points in the resulting p(r) curve
    :return: evidence, c, p(r), r, radius where
    """
    cdef:
        double[::1] variance
        double I_max
        int size_q, j 
        
        
    size_q = q.size
    assert size_q == I.size, "Intensity array has proper size"
    assert size_q == I_std.size, "Intensity array has proper size"
    
    
    variance = err**2

    p, r = makePriorDistribution(i[0], N, dmax) #Note, here I use p for what Hansen calls m
    T = createTransMatrix(q, r)

    p[0] = 0
    f = np.zeros_like(p)

    norm_T = T/err[:,None]  #Slightly faster to create this first

    sum_dia = np.sum(norm_T*i[:,None], axis=0)   #Creates YSUM in BayesApp code, some kind of calculation intermediate
    sum_dia[0] = 0

    B = np.dot(T.T, norm_T)     #Creates B(i, j) in BayesApp code
    B[0,:] = 0
    B[:,0] = 0

    #Do some kind of rescaling of the input
    c1 = np.sum(np.sum(T[1:4,1:-1]*p[1:-1], axis=1)/err[1:4])
    c2 = np.sum(i[1:4]/err[1:4])
    p[1:-1] = p[1:-1]*(c2/c1)
    f[1:-1] = p[1:-1]*1.001     #Note: f is called P in the original RAW BIFT code

    # Do the optimization
    t0 = time.perf_counter()
    tmp = bift_inner_loop(f, p, B, alpha, N, sum_dia)
    timings.append(time.perf_counter()-t0)
#     print(tmp)
    f, p, sigma, dotsp, xprec = tmp

    # Calculate the evidence
    s = np.sum(-(f[1:-1]-p[1:-1])**2/sigma[1:-1])
    c = np.sum((i[1:-1]-np.sum(T[1:-1,1:-1]*f[1:-1], axis=1))**2/err[1:-1])/i.size

    u = np.sqrt(np.abs(np.outer(f[1:-1], f[1:-1])))*B[1:-1, 1:-1]/alpha
    u[np.diag_indices(u.shape[0])] = u[np.diag_indices(u.shape[0])]+1
    w = np.linalg.svd(u, compute_uv = False)
    rlogdet = np.sum(np.log(np.abs(w)))

    evidence = -np.log(abs(dmax))+(alpha*s-0.5*c*i.size)-0.5*rlogdet-np.log(abs(alpha))

    # Some kind of after the fact adjustment

    if evidence <= 0 and dotsp < xprec:
        evidence=evidence*30
    elif dotsp < xprec:
        evidence = evidence/30.

    return evidence, c, f, r

