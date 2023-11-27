#Cython module to calculate distances of set of atoms

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "2023, ESRF"

import sys
import cython
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


cdef inline floating hard_sphere(floating pos, floating radius) noexcept nogil:
    """Density using hard spheres
    @param pos: fabs(d1-d)
    """
    if pos > 2.0 * radius:
        return 0.0 
    return (4 * radius + pos) * (2 * radius - pos) ** 2 / (16.0 * radius ** 3)

cdef inline floating soft_sphere(floating pos, floating radius) noexcept nogil:
    """Density using soft spheres (gaussian density)
    @param pos: fabs(d1-d)
    @param radius: radius of the equivalent hard sphere
    """
    cdef floating sigma = 0.40567 * radius
    return exp(- pos * pos / (2.0 * sigma * sigma)) * 0.3989422804014327 / sigma


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
        double d, d2, dx, dy, dz, x1, y1, z1
        double s1=0.0, s2=0.0, big = sys.maxsize
        double[:] min_col = numpy.zeros(size2, numpy.float64) + big
    assert atoms1.shape[1] >= 3
    assert atoms2.shape[1] >= 3
    
    for i in range(size1):
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
            min_col[j] = min(min_col[j], d2) 
        s1 += d
    for j in range(size2):
        s2 += min_col[j]
    
    return sqrt(0.5*((1./(size1*fineness2*fineness2))*s1 + (1./(size2*fineness1*fineness1))*s2))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calc_density(floating[:, :] atoms, floating dmax, int npt, floating r=0.0, bint hard=True):
    """
    Calculate the density rho(r)
    
    #TODO: formula for rigid sphere:
    A = (4*R+d)*(2*R-d)**2/16.0/R**3
    
    @param atoms: 2d-array with atom coordinates[[x,y,z],...]
    @param dmax: Diameter of the model
    @param npt: number of point in the density
    @param r: radius of an atom
    @param hard: use hard spheres model
    @return: 1d-array of 
    """
    
    cdef:
        int i, j, k, size = atoms.shape[0]        
        int width = 1 if hard else 2 
        floating d, dmax_plus, dx, dy, dz, x1, y1, z1
        floating delta, d_min, d_max, d1, den 
        double[::1] out = numpy.zeros(npt, numpy.float64)
        double s
        
    assert atoms.shape[1] >= 3
    assert size > 0
    assert dmax > 0
    dmax_plus = dmax * (1.0 + numpy.finfo(numpy.float32).eps)
    delta = dmax_plus / npt
    
    for i in range(size, nogil=True):
        x1 = atoms[i, 0]
        y1 = atoms[i, 1]
        z1 = atoms[i, 2]
        for j in range(size):
            dx = atoms[j, 0] - x1
            dy = atoms[j, 1] - y1
            dz = atoms[j, 2] - z1
            d = sqrt(dx * dx + dy * dy + dz * dz)
            d_min = max(0.0, d - width * r)
            d_max = min(dmax, d + width * r)
            for k in range(<int>(d_min / delta), <int>(d_max / delta)+1):
                if hard:
                    out[k] += hard_sphere(fabs(k * delta - d), r)
                else:
                    out[k] += soft_sphere(fabs(k * delta - d), r)
    return numpy.asarray(out)

