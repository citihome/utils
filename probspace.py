# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 21:29:35 2016

@author: hutch
"""

import numpy
    
def euclid2prob(L2_x, P_x, offset):
    """corresponding elements of probability space
    """
    L2_x = numpy.clip(L2_x-L2_x.max(), -offset, 0)
    exp_x = numpy.exp(L2_x)
    if P_x is None:
        return exp_x/exp_x.sum()
    else:
        P_x[:] = exp_x/exp_x.sum()

def euclid2logic(L2_x, offset):
    """corresponding elements of sign and its dual space w.r.t that of euclidean space
    """
    #for overflow instance
    P_x = (L2_x > 0).astype("float32")
    Pasterik_x = numpy.zeros_like(L2_x)
    
    #for instance with valid range
    valid = L2_x > -offset
    normalizer = 1+numpy.exp(-L2_x[valid])
    P_x[valid] = 1./normalizer
    Pasterik_x[valid] = L2_x[valid]+numpy.log(normalizer)#x+log(1+exp(-theta))
    
    return P_x, Pasterik_x
    
    
def euclid2sign(L2_x, offset):
    """corresponding elements of sign and its dual space w.r.t that of euclidean space
    """
    #for overflow instance
    P_x = numpy.sign(L2_x)
    Pasterik_x = L2_x.copy()
    
    #for instance with valid range
    valid = numpy.abs(L2_x) < offset
    normalizer = 1+numpy.exp(-L2_x[valid])
    P_x[valid] = 2./normalizer-1
    Pasterik_x[valid] += 2*numpy.log(normalizer)#x+2*log(1+exp(-theta))
    
    return P_x, Pasterik_x
    
def stickbreaking(L2_x, offset):    
    Pasterik = numpy.hstack([euclid2logic(L2_x, offset)[1], numpy.zeros((L2_x.shape[0],1))])
    Pasterik[:, 1:] += numpy.cumsum(Pasterik[:, :-1]-L2_x,1)
    return Pasterik