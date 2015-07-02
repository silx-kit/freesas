#Cython module to calculate distances of set of atoms

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import sys
import cython
cimport numpy
import numpy
from cython cimport floating
from libc.math cimport sqrt
from cython import parallel
cimport openmp 


@cython.wraparound(False)
@cython.boundscheck(False)
def calc_invariants(floating[:, :] atoms):
    """
    Calculate the fineness of the structure, i.e the average distance between the neighboring points in the model.

    Nota: to economize size*numpy.sqrt, the sqrt is taken at the end of the calculation. 
    We should have done s += sqrt(d) and then s/size, but we do s+= d and then sqrt(s/size).
    You can check that the result is the same.

    @param atoms: 2d-array with atom coordinates:[[x,y,z],...]
    @return: 3-tuple containing (fineness, Rg, Dmax)
        * average distance between an atoms and its nearest neighbor
        * radius of giration of the model
        * diameter of the model
    """
    cdef:
        int i, j, size
        floating d, x1, y1, z1, dx, dy, dz, big, d2, sum_d2, d2max
    size = atoms.shape[0]
    assert atoms.shape[1] >= 3
    big = sys.maxsize
    s = 0.0
    sum_d2 = 0.0 
    d2max = 0.0
    for i in range(size):
        x1 = atoms[i, 0]
        y1 = atoms[i, 1]
        z1 = atoms[i, 2]
        d = big
        for j in range(size):
            if i == j:
                continue
            dx = atoms[j, 0] - x1
            dy = atoms[j, 1] - y1
            dz = atoms[j, 2] - z1
            d2 = dx * dx + dy * dy + dz * dz
            sum_d2 += d2
            d2max = max(d2max, d2)
            d = min(d, d2)
        s += d
    return sqrt(s / size), sqrt(sum_d2 / 2.0) / size, sqrt(d2max)


@cython.wraparound(False)
@cython.boundscheck(False)
def calc_distance(floating[:, :] atoms1, floating[:, :] atoms2, floating fineness1, floating fineness2):
    """
    Calculate the Normalized Spatial Discrepancy (NSD) between two molecules
    
    @param atoms1,atoms2: 2d-array with atom coordinates[[x,y,z],...]
    @param fineness1, fineness2: fineness of each molecule
    @return: NSD atoms1-atoms2
    """
    
    cdef:
        int i, j, size1 = atoms1.shape[0], size2 = atoms2.shape[0]
        int threadid, numthreads = openmp.omp_get_max_threads() 
        double d, d2, dx, dy, dz, x1, y1, z1
        double s1 = 0.0, s2 = 0.0, big = sys.maxsize
        double[:, ::1] min_col = numpy.zeros((numthreads, size2), numpy.float64) + big
    assert atoms1.shape[1] >= 3
    assert atoms2.shape[1] >= 3
    assert size1 > 0
    assert size2 > 0
    
    for i in parallel.prange(size1, nogil=True):
        threadid = parallel.threadid()
        x1 = atoms1[i, 0]
        y1 = atoms1[i, 1]
        z1 = atoms1[i, 2]
        d = big
        for j in range(size2):            
            dx = atoms2[j, 0] - x1
            dy = atoms2[j, 1] - y1
            dz = atoms2[j, 2] - z1
            d2 = dx * dx + dy * dy + dz * dz
            d = min(d, d2)
            min_col[threadid, j] = min(min_col[threadid, j], d2) 
        s1 += d
    for j in parallel.prange(size2, nogil=True):
        d = big
        for i in range(numthreads):
            d = min(d, min_col[i, j])
        s2 += d

    return sqrt(0.5 * ((1.0 / (size1 * fineness2 * fineness2)) * s1 + (1.0 / (size2 * fineness1 * fineness1)) * s2))


@cython.wraparound(False)
@cython.boundscheck(False)
def calc_density(floating[:, :] atoms, floating dmax, int npt):
    """
    Calculate the density rho(r)
    
    @param atoms: 2d-array with atom coordinates[[x,y,z],...]
    @param dmax: Diameter of the model
    @param npt: number of point in the density
    @return: 1d-array of 
    """
    
    cdef:
        int i, j, k, size = atoms.shape[0]
        numpy.uint32_t s
        int threadid, numthreads = openmp.omp_get_max_threads() 
        floating d, dmax_plus, dx, dy, dz, x1, y1, z1
        floating s1 = 0.0, s2 = 0.0, big = sys.maxsize
        numpy.uint32_t[:, ::1] tmp = numpy.zeros((numthreads, npt), numpy.int64)
        numpy.uint32_t[::1] out = numpy.zeros(npt, numpy.int64)
        
    assert atoms.shape[1] >= 3
    assert size > 0
    dmax_plus = dmax * (1.0 + numpy.finfo(numpy.float32).eps)
    for i in parallel.prange(size, nogil=True):
        threadid = parallel.threadid()
        x1 = atoms[i, 0]
        y1 = atoms[i, 1]
        z1 = atoms[i, 2]
        for j in range(i):
            dx = atoms[j, 0] - x1
            dy = atoms[j, 1] - y1
            dz = atoms[j, 2] - z1
            d = sqrt(dx * dx + dy * dy + dz * dz)
            k = <int> (npt * d / dmax_plus)
            if k >= npt: 
                continue
            tmp[threadid, k] += 2 
        tmp[threadid, 0] += 1

    for j in parallel.prange(size, nogil=True):
        s = 0
        for i in range(numthreads):
            s = s + tmp[i, j]
        out[j] += s

    return out
