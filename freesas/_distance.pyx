#Cython module to calculate distances of set of atoms

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import sys
import cython
cimport numpy
import numpy
from cython cimport floating
from libc.math cimport sqrt, fabs, exp


@cython.wraparound(False)
@cython.boundscheck(False)
def calc_invariants(floating[:, :] atoms):
    """
    Calculate the invariants of the structure, i.e fineness, radius of gyration and diameter of the model.

    Nota: to economize size*numpy.sqrt, the sqrt is taken at the end of the calculation. 
    We should have done s += sqrt(d) and then s/size, but we do s+= d and then sqrt(s/size).
    You can check that the result is the same.

    @param atoms: 2d-array with atom coordinates:[[x,y,z],...]
    @return: 3-tuple containing (fineness, Rg, Dmax)
        * average distance between an atoms and its nearest neighbor
        * radius of gyration of the model
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


cdef inline floating hard_sphere(floating pos, floating radius)nogil:
    """Density using hard spheres
    @param pos: fabs(d1-d)
    """
    if pos > 2.0 * radius:
        return 0.0 
    return (4 * radius + pos) * (2 * radius - pos) ** 2 / (16.0 * radius ** 3)

cdef inline floating soft_sphere(floating pos, floating radius)nogil:
    """Density using soft spheres (gaussian density)
    @param pos: fabs(d1-d)
    @param radius: radius of the equivalent hard sphere
    """
    cdef floating sigma = 0.40567 * radius
    return exp(- pos * pos / (2.0 * sigma * sigma)) * 0.3989422804014327 / sigma


IF HAVE_OPENMP:
    include "_distance_omp.pxi"
ELSE:
    include "_distance_nomp.pxi"
