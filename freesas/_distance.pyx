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
def calc_fineness(floating[:,:] atoms):
    """
    Calculate the fineness of the structure, i.e the average distance between the neighboring points in the model.
    nota: to economize size*numpy.sqrt, the sqrt is taken at the end of the calculation. We should have done
    s += sqrt(d) and then s/size, but we do s+= d and then sqrt(s/size).
    You can check that the result is the same.

    @param atoms: 2d-array with atom coordinates:[[x,y,z],...]
    @return: average distance between an atoms and its nearest neighbor
    """
    cdef:
        int i, j, size
        floating d, x1, y1, z1, dx, dy, dz, big
    size = atoms.shape[0]
    assert atoms.shape[1] >= 3
    big = sys.maxsize
    s = 0.0
    for i in range(size):
        x1 = atoms[i,0]
        y1 = atoms[i,1]
        z1 = atoms[i,2]
        d = big
        for j in range(0, size):
            if i==j:
                continue
            dx = atoms[j,0] - x1
            dy = atoms[j,1] - y1
            dz = atoms[j,2] - z1
            d = min(d, dx*dx + dy*dy + dz*dz)
        s += d
    return sqrt(s/size)

@cython.wraparound(False)
@cython.boundscheck(False)
def calc_distance(floating[:,:] atoms1, floating[:,:] atoms2, floating fineness1, floating fineness2):
    """
    Calculate the Normalized Spatial Discrepancy (NSD) between two molecules
    
    @param atoms1,atoms2: 2d-array with atom coordinates[[x,y,z],...]
    @param fineness1, fineness2: fineness of each molecule
    @return: NSD atoms1-atoms2
    """
    
    
    cdef:
        int i, j, size1=atoms1.shape[0], size2=atoms2.shape[0]
        int threadid, numthreads = openmp.omp_get_max_threads() 
        double d, d2, dx, dy, dz, x1, y1, z1
        double s1=0.0, s2=0.0, big = sys.maxsize
        double[:,:] min_col = numpy.zeros((numthreads, size2), numpy.float64) + big
    assert atoms1.shape[1] >= 3
    assert atoms2.shape[1] >= 3
    
    for i in parallel.prange(size1, nogil=True):
        threadid = parallel.threadid()
        x1 = atoms1[i,0]
        y1 = atoms1[i,1]
        z1 = atoms1[i,2]
        d = big
        for j in range(size2):            
            dx = atoms2[j,0] - x1
            dy = atoms2[j,1] - y1
            dz = atoms2[j,2] - z1
            d2 = dx*dx + dy*dy + dz*dz
            d = min(d, d2)
            min_col[threadid, j] = min(min_col[threadid, j], d2) 
        s1 += d
    for j in parallel.prange(size2, nogil=True):
        d = big
        for i in range(numthreads):
            d = min(d, min_col[i, j])
        s2 += d

    return sqrt(0.5*((1./(size1*fineness2*fineness2))*s1 + (1./(size2*fineness1*fineness1))*s2))
