__author__ = "Jerome Kieffer"
__license__ = "MIT"
__copyright__ = "2017, ESRF"

import numpy
from cython cimport floating


class LongestRunOfHeads(object):
    """Implements the "longest run of heads" by Mark F. Schilling
    The College Mathematics Journal, Vol. 21, No. 3, (1990), pp. 196-207
    
    See: http://www.maa.org/sites/default/files/pdf/upload_library/22/Polya/07468342.di020742.02p0021g.pdf
    """
    def __init__(self):
        "We store already calculated values for (n,c)"
        self.knowledge = {}

    def A(self, int n, int c):
        """Calculate A(number_of_toss, length_of_longest_run)
        
        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of 
        :return: The A parameter used in the formula
        
        """
        if n <= c:
            return 1 << n
        elif (n, c) in self.knowledge:
            return self.knowledge[(n, c)]
        else:
            s = 0
            for j in range(c, -1, -1):
                s += self.A(n - 1 - j, c)
            self.knowledge[(n, c)] = s
            return s

    def __call__(self, int n, int c):
        """Calculate the probability of this to occur 
        
        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of heads, an integer 
        :return: The probablility of having c subsequent heads in a n toss of fair coin
        """
        return 1.0 - self.A(n, c) / 2.0 ** n


LROH = LongestRunOfHeads()


cpdef int measure_longest(floating[::1] ary):
    """Measure the longest region where ary>0 or ary<0
    
    :param ary: numpy array or list of floats
    :return: the longest region with same sign
    """
    cdef: 
        int last = 0
        int longest = 0
        int acc = 0
        int i, d
        floating v
        
    for i in range(ary.size):
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
    return longest


def gof(data1, data2):
    """Calculate the probability for a couple of dataset to be equivalent according to:
    http://www.nature.com/nmeth/journal/v12/n5/full/nmeth.3358.html
    
    :param data1: numpy array
    :param data2: numpy array
    :return: probablility for the 2 data to be equivalent
    """
    cdef:
        double[::1] cdata
        int c
    if data1.ndim == 2 and data1.shape[1] > 1:
        data1 = data1[:, 1]
    if data2.ndim == 2 and data2.shape[1] > 1:
        data2 = data2[:, 1]

    cdata = numpy.ascontiguousarray(data2 - data1, numpy.float64).ravel() 
    c = measure_longest(cdata)
    return LROH(cdata.size, c)
