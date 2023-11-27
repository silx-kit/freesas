__author__ = "Jerome Kieffer"
__license__ = "MIT"
__copyright__ = "2017, ESRF"

import numpy
from math import log
from .collections import GOF

from ._cormap import measure_longest


class LongestRunOfHeads(object):
    """Implements the "longest run of heads" by Mark F. Schilling
    The College Mathematics Journal, Vol. 21, No. 3, (1990), pp. 196-207
    
    See: http://www.maa.org/sites/default/files/pdf/upload_library/22/Polya/07468342.di020742.02p0021g.pdf
    """

    def __init__(self):
        "We store already calculated values for (n,c)"
        self.knowledge = {}

    def A(self, n, c):
        """Calculate A(number_of_toss, length_of_longest_run)
        
        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of 
        :return: The A parameter used in the formula
        
        """
        if n <= c:
            return 2 ** n
        elif (n, c) in self.knowledge:
            return self.knowledge[(n, c)]
        else:
            s = 0
            for j in range(c, -1, -1):
                s += self.A(n - 1 - j, c)
            self.knowledge[(n, c)] = s
            return s

    def B(self, n, c):
        """Calculate B(number_of_toss, length_of_longest_run)
        to have either a run of Heads either a run of Tails
        
        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of 
        :return: The B parameter used in the formula
        """
        return 2 * self.A(n - 1, c - 1)

    def __call__(self, n, c):
        """Calculate the probability for the longest run of heads to exceed the observed length  
        
        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of heads, an integer 
        :return: The probablility of having c subsequent heads in a n toss of fair coin
        """
        if c >= n:
            return 0
        delta = 2 ** n - self.A(n, c)
        if delta <= 0:
            return 0
        return 2.0 ** (log(delta, 2) - n)

    def probaHeadOrTail(self, n, c):
        """Calculate the probability of a longest run of head or tails to occur 
        
        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of heads or tails, an integer 
        :return: The probablility of having c subsequent heads or tails in a n toss of fair coin
        """
        if c > n:
            return 0
        if c == 0:
            return 0
        delta = self.B(n, c) - self.B(n, c - 1)
        if delta <= 0:
            return 0
        return min(2.0 ** (log(delta, 2.0) - n), 1.0)

    def probaLongerRun(self, n, c):
        """Calculate the probability for the longest run of heads or tails to exceed the observed length  
        
        :param n: number of coin toss in the experiment, an integer
        :param c: length of thee observed run of heads or tails, an integer 
        :return: The probablility of having more than c subsequent heads or tails in a n toss of fair coin
        """
        if c > n:
            return 0
        if c == 0:
            return 0
        delta = (2 ** n) - self.B(n, c)
        if delta <= 0:
            return 0
        return min(2.0 ** (log(delta, 2.0) - n), 1.0)


LROH = LongestRunOfHeads()


def gof(data1, data2):
    """Calculate the probability for a couple of dataset to be equivalent 
    
    Implementation according to:
    http://www.nature.com/nmeth/journal/v12/n5/full/nmeth.3358.html
    
    :param data1: numpy array
    :param data2: numpy array
    :return: probablility for the 2 data to be equivalent
    """

    if data1.ndim == 2 and data1.shape[1] > 1:
        data1 = data1[:, 1]
    if data2.ndim == 2 and data2.shape[1] > 1:
        data2 = data2[:, 1]

    cdata = numpy.ascontiguousarray(data2 - data1, numpy.float64).ravel()
    c = measure_longest(cdata)
    n = cdata.size
    res = GOF(n, c, LROH.probaLongerRun(n, c - 1))
    return res
