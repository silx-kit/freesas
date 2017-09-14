
from __future__ import print_function
import h5py
import numpy

class HPLCrun:
    
    sum_I = None
    runLength =  None
    
    def __init__(self, h5file = None):
        if h5file:         
            with h5py.File(h5file, "r") as data:
                self.q = numpy.asarray(data['q'])[:]
                self.I = numpy.asarray(data['scattering_I'])
                self.Ierr = numpy.asarray(data['scattering_Stdev'])
                self.runLength = self.I.shape[0]
                if "sum_I" in data.keys():
                    self.sum_I = numpy.asarray(data['sum_I'])
                self.trim()
                    
    def trim(self):
        """
        Remove data points of 0 signal from sum_I so that they don't srew up plotting
        """
        if self.sum_I is not None:
            self.sum_I[numpy.where(self.sum_I == 0)] = numpy.nan
            