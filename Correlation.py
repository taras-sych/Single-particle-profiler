import numpy as np
import math

# -*- coding: utf-8 -*-

# Kindly provided by Falk Schneider
    # Stolen from Dominic's GitHub. 
    # https://github.com/dwaithe/generalMacros/blob/master/diffusion%20simulations%20/Brownian%20Motion%20simulation%20End-to-End.ipynb
    # Edited for Falk's purposes :) 
    # Given to Franco for his enjoyment :)
    # And also to Agnes :) 
    # And now also to Taras

    # Falk Schneider 20/04/2020



def correlate_linear(a, b):
    """Return linear correlation of two vectors using DFT."""
    size = a.shape[0]
    
    # subtract mean and pad with zeros to twice the size
    a_mean = np.average(a)
    b_mean = np.average(b)
    
    #Has the padding in
    a = np.pad(a-a_mean, a.size//2, mode='constant')
    b = np.pad(b-b_mean, b.size//2, mode='constant')
    
    # forward DFT
    a = np.fft.rfft(a)
    b = np.fft.rfft(b)
    # multiply by complex conjugate
    c = a * b.conj()
    # reverse DFT
    c = np.fft.irfft(c)
    #positive delays only
    c = c[:size // 2]
        
    # normalize with the averages of a and b
    c /= size * a_mean * b_mean
    
    return np.array(c)

def binaver(c, bins):
    """Return averaged chunks of vector."""
    out = [np.average(c[:bins[0]])]
    for i in range(len(bins)-1):
        out.append(np.mean(c[bins[i]:bins[i+1]]))
    return np.array(out)


def logbins(size, nbins):
    """Return up to nbins exponentially increasing integers from 1 to size."""
    b = np.logspace(0, math.log(size, 2), nbins, base=2, endpoint=True)
    return np.unique(b.astype('intp'))


def smooth(c):
    """Return double exponentially smoothed vector."""
    out = c.copy()
    out[0] = out[1]
    for i in range(1, len(out)):
        out[i] = out[i] * 0.3 + out[i-1] * 0.7
    for i in range(len(out)-2, -1, -1):
        out[i] = out[i] * 0.3 + out[i+1] * 0.7
    return np.array(out)






