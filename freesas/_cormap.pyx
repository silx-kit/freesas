# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, cdivision=True, embedsignature=True

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__copyright__ = "2017, ESRF"

from cython cimport floating
from libc.stdlib cimport abs
import numpy 


cpdef int measure_longest(floating[::1] ary):
    """Measure the longest region where ary>0 or ary<0
    
    :param ary: numpy array or list of floats
    :return: the longest region with same sign
    """
    cdef: 
        int last = 0
        int longest = 0
        int acc = 0
        int i, d, size
        floating v
    size = ary.size
    with nogil:
        for i in range(size):
            v = ary[i]
            if v > 0:
                d = 1
            elif v < 0:
                d = -1
            else:
                d = 0
            if d * last <= 0:
                if abs(acc) > longest:
                    longest = abs(acc)
                acc = d
            else:
                acc = acc + d
            last = d
        if abs(acc) > longest:
            longest = abs(acc) 
    return longest
