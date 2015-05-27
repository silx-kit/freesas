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

@cython.wraparound(False)
@cython.boundscheck(False)
def calc_fineness(floating[:,:] atoms):
    """
    Calculate the fineness of the structure, i.e the average distance between the neighboring points in the model

    @param atoms: 2d-array with atom coordinates:[[x,y,z],...]
    @return: average distance between an atoms and its nearest neighbor
    """
    cdef:
        int i, j, size
        floating d, x1, y1, z1, dx, dy, dz, big
    size = atoms.shape[0]
#     assert atoms.shape[1] == 3
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
