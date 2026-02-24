#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file contains several toolbox functions to process 1D numpy arrays - basic signal treatment, filters, etc.
# 
# Created by R. Proux, 16/10/2017

import pylab as pl
import scipy.signal

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Copied and modified from http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html  (16/10/2017 - RProux)
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: the window_len parameter should always be odd for reliable output (will shift slightly the data if even)
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = pl.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    print(s)
    print(len(s))
    if window == 'flat': #moving average
        w = pl.ones(window_len, 'd')
    else:
        w = eval('pl.{}(window_len)'.format(window))

    y = pl.convolve(w / w.sum(), s, mode='valid')

    return y[int(pl.floor(window_len/2)) : -int(pl.ceil(window_len/2) - 1)]


def filter_cosmic_rays(spectrum, error_thr=10., filter_size=5):
    """
    Filters out cosmic rays from a 1D spectrum (simple spike detection algorithm).
    
    Args:
        spectrum (numpy array): a 1D numpy array with spectrum data
        error_thr (float, optional): spike detection threshold. If the distance from smoothened spectrum to real 
        spectrum is greater than error_thr, data will be replaced by smoothened spectrum.
        filter_size (int, optional): number of pixels of the smoothening window.
    
    Returns:
        numpy array: the spectrum corrected (spikes removed).
    """

    spectrum_smooth = scipy.signal.medfilt(spectrum, filter_size)
    bad_pixels = pl.absolute(spectrum - spectrum_smooth) > float(error_thr)
    spectrum_corr = spectrum.copy()
    spectrum_corr[bad_pixels] = spectrum_smooth[bad_pixels]
    
    return spectrum_corr

    
def range_to_edge(middles):
    """Converts from a list of values to list of boundaries
    i.e. (1, 2, 3) becomes (0.5, 1.5, 2.5, 3.5)
    Useful for pcolorfast plotting.
    Can handle irregular values (not evenly spaced). It uses the middle of each interval 
    and adds half an interval at the beginning and at the end.
    
    Args:
        middles (iterable): 1D list of numbers (can be a numpy array or a list)
    
    Returns:
        numpy array: list of edges (has len(middles)+1 elements)
    """
    edges = [(float(leftMiddle) + rightMiddle) / 2 for leftMiddle, rightMiddle in zip(middles[1:], middles[:-1])]
    edges.insert(0, float(middles[0]) - (edges[0] - middles[0]))
    edges.append(float(middles[-1]) + (middles[-1] - edges[-1]))
    
    return pl.array(edges)


def crop_array(array, top, right, bottom, left):
    """Crops a 2D numpy array
    
    Args:
        array (numpy array): array to crop
        top (int): number of pixels to crop at the top of the array
        right (int): number of pixels to crop at the right of the array
        bottom (int): number of pixels to crop at the bottom of the array
        left (int): number of pixels to crop at the left of the array
    
    Returns:
        numpy array: cropped array
    """
    return array[top:-bottom, left:-right]


def histeq(im, nb_bins=256):
    """
    This function equalises the histogram of im (numpy array), distributing
    the pixels accross nbr_bins bins.
    Returns the equalised image as a numpy array (same shape as input).
    Largely copied from https://stackoverflow.com/a/28520445  
        (author: Trilarion, 17/07/2017)
    """
    # get image histogram
    imhist, bins = pl.histogram(im.flatten(), nb_bins, normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 65535 * cdf / cdf[-1] #normalize
    
    # use linear interpolation of cdf to find new pixel values
    im2 = pl.interp(im.flatten(), bins[:-1], cdf)
    
    return im2.reshape(im.shape)


if __name__ == '__main__':
    x = pl.linspace(0, 10, 100)
    data = pl.sin(x)

    data_smooth = smooth(data, 13)
    
    pl.figure()
    pl.plot(x, data)
    pl.plot(x, data_smooth)
    pl.plot(x, scipy.signal.medfilt(data, 5))
    pl.show()
