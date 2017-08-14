# -*- coding: utf-8 -*-
from __future__ import division, print_function
__author__ = "Martha Brennich"
__license__ = "MIT"
__copyright__ = "2017, EMBL"

"""
Loosely based on the autoRg implementation in BioXTAS RAW by J. Hopkins
"""





class Error(Exception):
    """Base class for exceptions in this module."""
pass

class InsufficientDataError(Error):
    """Exception raised for errors in the input.
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self):
        self.expression = ""
        self.message = "Not enough data do determine Rg"


#from numpy import sqrt as nsqrt#from numpy import exp as nexp
cimport numpy as cnp
import numpy  as np
from math import exp

cdef extern from "math.h":
    float sqrt(float m)
    #float exp(float m) #Note: For exp, python math seems faster?

DTYPE = np.float
ctypedef cnp.float_t DTYPE_t

cdef float qmaxrg_weight = 1
cdef float qminrg_weight = 0.1
cdef float rg_frac_err_weight = 1
cdef float i0_frac_err_weight = 1
cdef float r_sqr_weight = 4
cdef float reduced_chi_sqr_weight = 0
cdef float window_size_weight = 6

cdef cnp.ndarray weights = np.array([qmaxrg_weight, qminrg_weight, rg_frac_err_weight, i0_frac_err_weight, r_sqr_weight,
                    reduced_chi_sqr_weight, window_size_weight])


def weightedlinFit(cnp.ndarray x, cnp.ndarray y, cnp.ndarray w):
    """Calcualtes a fit to a - bx, weighted by w. 
        Input:
        x, y: The dataset to be fitted.
        w: The weight fot the individual points in x,y. Typically w would be 1/yerr**2.
        Returns: xopt = (a,b) and dx = (da,db)
    """
    
    cdef cnp.ndarray datax = x
    cdef cnp.ndarray datay = y
    cdef cnp.ndarray weight = w
    cdef int n = datax.shape[0]
    cdef float sigma = weight.sum()
    cdef cnp.ndarray yw = datay*weight
    cdef cnp.ndarray dxx = datax*datax
    cdef float sigmay = yw.sum()#(datay*weight).sum() #This is faster then calculating the inner product
    cdef float sigmax = (datax*weight).sum()
    cdef float sigmaxy = (datax*yw).sum()#(datax*datay*weight).sum()
    cdef float sigmaxx = (dxx*weight).sum()#(datax*datax*weight).sum()
        #x = [-sigmaxx*sigmay + sigmax*sigmaxy,-sigmax*sigmay+sigma*sigmaxy]/detA
    cdef float a = 0 
    cdef float b= 0 
    cdef float xmean = 0 
    cdef float ymean = 0 
    cdef float ssxx= 0 
    cdef float ssyy = 0 
    cdef float ssxy= 0 
    cdef float s =  0 
    cdef float da = 0 
    cdef float db = 0 
    cdef float xmean2 = 0
    cdef float detA = sigmax*sigmax-sigma*sigmaxx
    if not(detA==0):
        #x = [-sigmaxx*sigmay + sigmax*sigmaxy,-sigmax*sigmay+sigma*sigmaxy]/detA
        a = (-sigmaxx*sigmay + sigmax*sigmaxy)/detA
        b = (-sigmax*sigmay+sigma*sigmaxy)/detA
        xmean = datax.mean()
        ymean = datay.mean()
        xmean2 = xmean*xmean
        ssxx = (dxx).sum() -n*xmean2#*xmean
        ssyy = (datay*datay).sum() -n*ymean*ymean
        #ssxy = (datay*datax).sum() -n*ymean*xmean
        s = sqrt((ssyy + b*b*ssxx)/(n-2))
        da = sqrt(s*(1/n + (xmean2)/ssxx))
        db = sqrt(s/ssxx)
        xopt = (a,b)
        dx = (da,db)
        return xopt, dx

def linear_func(cnp.ndarray x,float a, float b):
    #return a-b*x
    return np.add(a,-b*x)



def autoRg(cnp.ndarray sasm):
    """This function automatically calculates the radius of gyration and scattering intensity at zero angle
    from a given scattering profile. It roughly follows the method used by the autorg function in the atsas package
    Input:
    sasm: An array of q, I(q), dI(q)
    """
    sasm = sasm[np.where((np.isnan(sasm[:,2]) == False)*(np.isinf(sasm[:,2]) == False)*(sasm[:,2] > 0))]
  
    #We will define q and ierr later, to avoid unnecessary assignments
    cdef cnp.ndarray i = sasm[:,1]

    cdef cnp.ndarray quality 


    cdef int qmin = 0
    cdef int qmax = -1

  

    #Pick the start of the RG fitting range. Note that in autorg, this is done
    #by looking for strong deviations at low q from aggregation or structure factor
    #or instrumental scattering, and ignoring those. This function isn't that advanced
    #so we start at 0 or the first positive data point
    cdef int data_start = 0
    data_start = max(qmin,np.argmax(i > 0))

    i = i[data_start:-1]
   
    #Following the atsas package, the end point of our search space is the q value
    #where the intensity has droped by an order of magnitude from the initial value.
  
    cdef int data_end = 0
    data_end = np.argmax(abs(i) < abs(i[0]/10)) #This is deiffernt from waht Jesse does, but also works...

    if (data_end ) < 10:
        raise InsufficientDataError()
  
    #This makes sure we're not getting some weird fluke at the end of the scattering profile.
    if data_end > len(i)/2.:
        found = False
        idx = 0
        while not found:
            idx = idx +1
            if i[idx]<i[data_start]/10:
                found = True
            elif idx == len(i) -1:
                found = True
        data_end = idx
    #Let's assign q and ierr abd remove the trailing points from i
    cdef cnp.ndarray q = sasm[data_start:data_start+data_end,0]
    i = i[0:data_end]
    cdef cnp.ndarray err = sasm[data_start:data_start+data_end,2]

    #Start out by transforming: we need x: q*q, y: log I and w: (err/i)**(-2) 
    cdef cnp.ndarray qs = q*q
    cdef cnp.ndarray il = np.log(i)

    cdef cnp.ndarray ilerinv = i/err
    cdef cnp.ndarray ilw = ilerinv * ilerinv
    


    #Pick a minimum fitting window size. 10 is consistent with atsas autorg.
    min_window = 10

    max_window = data_end-data_start

    fit_list = []

    #It is very time consuming to search every possible window size and every possible starting point.
    #Here we define a subset to search.
    tot_points = max_window
    window_step = int(tot_points/10)
    data_step = int(tot_points/50)

    if window_step == 0:
        window_step =1
    if data_step ==0:
        data_step =1

    window_list = range(min_window,max_window+1, window_step)
 

    cdef cnp.ndarray x,y,yw, diff, diff2
    cdef int dof
    #cdef float RG, I0, RGer, I0er, a,b, r_sqr, chi_sqr
    #This function takes every window size in the window list, stepts it through the data range, and
    #fits it to get the RG and I0. If basic conditions are met, qmin*RG<1 and qmax*RG<1.35, and RG>0.1,
    #We keep the fit.
    for w in window_list:
        for start in range(data_start,data_end-w, data_step):
            x = qs[start:start+w]
            y = il[start:start+w]
            yw = ilw[start:start+w]

            

            try:
                opt,dopt = weightedlinFit(x, y, yw)
            except ValueError as VE:
                print(VE)
                raise 
            except Exception as err:
                print("An error occured. y = ", y, "yw = ", yw, "x = ", x)
                raise 
            else:
                a = opt[0]
                b = opt[1]
                da = dopt[0]
                db = dopt[1]
                lower = q[start]*q[start]*b
                upper = q[start+w-1]*q[start+w-1]*b
                if b>3*10**(-5) and lower <0.33 and upper<0.6075 and db/b <= 1:

                    a = opt[0]
                    b = opt[1]

                    #chi_sqr,r_sqr = chiR(opt,data)
                    diff = np.add(y,-linear_func(x, a, b))
                    diff2 = diff*diff
                    #diff2 = np.square(diff)
                    r_sqr = 1 - diff2.sum()/np.square(y-y.mean()).sum()
                    #r_sqr = 1 - diff2.sum()/((y-y.mean())*(y-y.mean())).sum()
                    if r_sqr > .15:
                        chi_sqr = (diff2*yw).sum()

                        #All of my reduced chi_squared values are too small, so I suspect something isn't right with that.
                        #Values less than one tend to indicate either a wrong degree of freedom, or a serious overestimate
                        #of the error bars for the system.
                        dof = w - 2.
                        reduced_chi_sqr = chi_sqr/dof

                        fit_list.append([start, w, q[start], q[start+w-1], b, db, a, da, lower, upper, r_sqr, chi_sqr, reduced_chi_sqr])
        #Extreme cases: may need to relax the parameters.
 
    if len(fit_list)<1:
        #Stuff goes here
        pass

    if len(fit_list)>0:
        fit_list = np.array(fit_list)

        #Now we evaluate the quality of the fits based both on fitting data and on other criteria.


        max_window_real = float(window_list[-1]) #To ensure float division in Python 2

        #all_scores = []
        qmaxrg_score = 1-np.absolute((fit_list[:,9]-0.56)/0.56)
        qminrg_score = 1-fit_list[:,8]
        rg_frac_err_score = 1-fit_list[:,5]/fit_list[:,4]
        i0_frac_err_score = 1 - fit_list[:,7]/fit_list[:,6]
        r_sqr_score = fit_list[:,10]
        reduced_chi_sqr_score = 1/fit_list[:,12] #Not right
        window_size_score = fit_list[:,1]/max_window_real
        scores = np.array([qmaxrg_score, qminrg_score, rg_frac_err_score, i0_frac_err_score, r_sqr_score,
                               reduced_chi_sqr_score, window_size_score])
        quality = np.dot(weights,scores)/weights.sum()
       
        #I have picked an aribtrary threshold here. Not sure if 0.6 is a good quality cutoff or not.
        if quality.max() > 0:# 0.5:
            # idx = quality.argmax()
            # rg = fit_list[idx,4]
            # rger1 = fit_list[idx,5]
            # i0 = fit_list[idx,6]
            # i0er = fit_list[idx,7]
            # idx_min = fit_list[idx,0]
            # idx_max = fit_list[idx,0]+fit_list[idx,1]

            # try:
            #     #This adds in uncertainty based on the standard deviation of values with high quality scores
            #     #again, the range of the quality score is fairly aribtrary. It should be refined against real
            #     #data at some point.
            #     rger2 = fit_list[:,4][quality>quality[idx]-.1].std()
            #     rger = rger1 + rger2
            # except:
            #     rger = rger1

            try:
                idx = quality.argmax()
                #rg = fit_list[:,4][quality>quality[idx]-.1].mean()
                
                rg = sqrt(3.*fit_list[idx,4])
                dber = fit_list[:,5][quality>quality[idx]-.1].std()
                rger = 0.5*sqrt(3./rg)*dber
                i0 = exp(fit_list[idx,6])
                #i0 = fit_list[:,6][quality>quality[idx]-.1].mean()
                daer = fit_list[:,7][quality>quality[idx]-.1].std()
                i0er = i0*daer
                idx_min = int(fit_list[idx,0])
                idx_max = int(fit_list[idx,0]+fit_list[idx,1]-1)
                idx_min_corr = np.argmin(np.absolute(sasm[:,0] - fit_list[idx,3]))
                idx_max_corr = np.argmin(np.absolute(sasm[:,0] - fit_list[idx,4]))
            except:
                idx = quality.argmax()
                rg = sqrt(3.*fit_list[idx,4])
                rger = 0.5*sqrt(3./rg)*fit_list[idx,5]
                i0 = exp(fit_list[idx,6])
                i0er = i0*fit_list[idx,7]
                idx_min = int(fit_list[idx,0])
                idx_max = int(fit_list[idx,0]+fit_list[idx,1]-1)


        else:
          
            rg = -1
            rger = -1
            i0 = -1
            i0er = -1
            idx_min = -1
            idx_max = -1

    else:
       
        rg = -1
        rger = -1
        i0 = -1
        i0er = -1
        idx_min = -1
        idx_max = -1
        #quality = []
        all_scores = []

    idx_min = idx_min + data_start
    idx_max = idx_max + data_start

    #We could add another function here, if not good quality fits are found, either reiterate through the
    #the data and refit with looser criteria, or accept lower scores, possibly with larger error bars.

    #returns Rg, Rg error, I0, I0 error, the index of the first q point of the fit and the index of the last q point of the fit
    return rg, rger, i0, i0er, idx_min, idx_max