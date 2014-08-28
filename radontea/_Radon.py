#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    _Radon.py
    
    Performs the radon transform.
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import scipy
import scipy.ndimage


__all__ = ["radon", "radon_fan_translation"]


def radon(arr, angles, callback=None, cb_kwargs={}):
    """ Compute the Radon transform (sinogram) of a circular image.


    The `scipy` Radon transform performs this operation on the 
    entire image, whereas this implementation requires an input
    image that has gray-scale values of 0 outside of a circle
    with diameter equal to the image size.


    Parameters
    ----------
    arr : ndarray, shape (N,N)
        the input image.
    angles : ndarray, length A
        angles or projections in radians
    callback : callable, optional
        If set, the function `callback` is called on a regular basis
        throughout this algorithm.
        Number of function calls: A+1
    cb_kwargs : dict, optional
        Keyword arguments for `callback` (e.g. "pid" of process).


    Returns
    -------
    outarr : ndarray of floats, shape (A,N)
        Sinogram of the input image. The i'th row contains the
        projection data of the i'th angle.


    See Also
    --------
    scipy.ndimage.interpolation.rotate :
        The interpolator used to rotate the image.
    """
    # This function also works with single angles
    angles = np.atleast_1d(angles)
    # The radon function from skimage.transform doeas not allow
    # to reshape the image (one could cut it of course).
    # Furthermore, no interpolation is used or at least I do not
    # know what kind of interpolation is used (_wharp_fast?).
    # outarray: x-axis: projection
    #           y-axis: 
    outarr = np.zeros((len(angles),len(arr)))
    if callback is not None:
        callback(**cb_kwargs)
    for i in np.arange(len(angles)):
        rotated = scipy.ndimage.rotate(arr, angles[i]/np.pi*180, order=3,
                  reshape=False, mode="constant", cval=0) #black corner
        # sum along some axis.
        outarr[i] = rotated.sum(axis=0)
        if callback is not None:
            callback(**cb_kwargs)
    return outarr


def radon_fan_translation(arr, det_size, lS=1, return_ang=False,
                          callback=None, cb_kwargs={}):
    """ Compute the Radon transform for a fan beam geometry
        
    In contrast to `radon`, this function uses (A) a fan-beam geometry
    (the integral is taken along rays that meet at one point), and
    (B) translates the object laterally instead of rotating it. The
    result is sometimes referred to as 'linogram'.

    x
    ^ 
    |
    ----> z


    source       object   detector

              /             . (., N/2)
    (0,-lS) ./   (0,0)      .
             \              .
              \             . (., N/2)


    The algorithm computes all angular projections for discrete
    movements of the object. The position of the object is changed such
    that its lower boundary starts at x=det_size/2 and ends at
    x=-(det_size/2+N) for odd det_size.


    Parameters
    ----------
    arr : ndarray, shape (N,N)
        the input image.
    det_size : int
        The total detector size in pixels. The detector centered to the
        source (det_size odd) or is moved half a pixel up (even). The 
        axial position of the detector is the center of the pixels on 
        the far right of the object.
    lS : int
        Source position relative to the center of leftest pixel of
        `arr`. lS >= 1.
    callback : callable, optional
        If set, the function `callback` is called on a regular basis
        throughout this algorithm.
        Number of function calls: N+det_size
    cb_kwargs : dict, optional
        Keyword arguments for `callback` (e.g. "pid" of process).


    Returns
    -------
    outarr : ndarray of floats, shape (A,N)
        Linogram of the input image. The i'th row contains the
        projection data of the i'th angle.


    See Also
    --------
    scipy.ndimage.interpolation.rotate :
        The interpolator used to rotate the image.
    radontea.radon
        The real radon transform.
    """
    N = len(arr)
    # First, create a zero-padded version of the input image such that
    # its center is the source.
    setup = np.pad(arr, ((0,0),(N+2*lS-1,0)), mode="constant")
    
    # Second, compute the rotational angles that we will need.
    if det_size%2 != 0: #odd
        x = np.linspace(-(det_size-1)/2, (det_size+1)/2, det_size,
                        endpoint=True)
    else: #even
        x = np.linspace(-det_size/2+1, det_size/2, det_size,
                        endpoint=True)
    angles = np.arctan2(x,N+lS-1)
    
    # Now we can rotate the image for every lateral position of the
    # object. We will again zero-pad the image with N+det_size values
    # and roll down the object.
    padset = np.pad(setup, ((0,N+det_size), (0,0)), mode="constant")
    
    lino = np.zeros((N+det_size, det_size))
    
    for i in range(N+det_size):
        print(i)
        padset = np.roll(padset, 1, axis=0)
        # cut out a det_size slice
        curobj = padset[N:N+det_size]
        for j in range(det_size):
            ang = angles[j]
            rotated = scipy.ndimage.rotate(curobj, ang/np.pi*180,
                        order=3, reshape=True, mode="constant", cval=0)
            if det_size%2 != 0: #odd
                centerid = int(np.floor(len(rotated)/2)+1)
            else: #even
                centerid = int(len(rotated)/2)
                
            lino[i,j] = np.sum(rotated[centerid, N+2*lS-1:])
            
        if callback is not None:
            callback(**cb_kwargs)
    if return_ang:
        return lino, angles
    else:
        return lino
