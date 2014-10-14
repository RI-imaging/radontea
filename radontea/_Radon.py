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



def radon_fan_translation(arr, det_size, det_spacing=1, shift_size=1,
                          lS=1, lD=None, return_ang=False,
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

              /             . (., lD)
    (0,-lS) ./   (0,0)      .
             \              .
              \             . (., lD)


    The algorithm computes all angular projections for discrete
    movements of the object. The position of the object is changed such
    that its center starts at (det_size/2, 0) and ends at
    (-det_size/2, 0) at increments of `shift_space`.


    Parameters
    ----------
    arr : ndarray, shape (N,N)
        the input image.
    det_size : int
        The total detector size in pixels. The detector centered to the
        source (det_size odd) or is moved half a pixel up (even). The 
        axial position of the detector is the center of the pixels on 
        the far right of the object.
    det_spacing : float
        Distance between detection points in pixels.
    shift_size : int
        The amount of pixels that the object is shifted between
        projections.
    lS : multiples of 0.5
        Source position relative to the center of `arr`. lS >= 1.
    lD : int
        Detector position relative to the center `arr`. Default is N/2.
    return_ang : bool
        Also return the angles corresponding to the detector pixels.
    callback : callable, optional
        If set, the function `callback` is called on a regular basis
        throughout this algorithm.
        Number of function calls: N+det_size
    cb_kwargs : dict, optional
        Keyword arguments for `callback` (e.g. "pid" of process).


    Returns
    -------
    outarr : ndarray of floats, shape (N+det_size,det_size)
        Linogram of the input image. Where N+det_size determines the
        lateral position of the sample.


    See Also
    --------
    scipy.ndimage.interpolation.rotate :
        The interpolator used to rotate the image.
    radontea.radon
        The real radon transform.
    """
    N = len(arr)
    if lD is None:
        lD = N/2
    lS = abs(lS)
    lDS = lS + lD
    
    # First, create a zero-padded version of the input image such that
    # its center is the source - this is necessary because we can
    # only rotate through the center of the image.
    #arrc = np.pad(arr, ((0,0),(N+2*lS-1,0)), mode="constant")    
    if lS <= N/2:
        pad = 2*lS
        arrc = np.pad(arr, ((0,0),(pad,0)), mode="constant")
        # The center is where either b/w 2 pixels or at one pixel,
        # as it is for the input image.
    else:
        pad = N/2 + lS
        if pad%1 == 0: #even
            pad = int(pad)
            # We padd to even number of pixels.
            # The center of the image is between two pixels (horizontally).
            arrc = np.pad(arr, ((0,0),(pad,0)), mode="constant")
        else: #odd
            # We padd to odd number of pixels.
            # The center of the image is a pixel.
            pad = int(np.floor(pad))
            arrc = np.pad(arr, ((0,0),(pad,0)), mode="constant")
    
    # Lateral and axial coordinates of detector pixels
    axi = np.ones(det_size)*lDS
    # minus .5 because we use the center of the pixel
    
    if det_size%2 == 0: #even
        latmax = det_size/2 * det_spacing
        lat = np.linspace(-latmax, latmax, det_size, endpoint=False)
    else: #odd
        latmax = det_size/2 * det_spacing - .5
        lat = np.linspace(-latmax, latmax, det_size, endpoint=True)

    angles = np.arctan2(lat,axi)
    #angles -= np.min(angles)
    
    
    # Before we can rotate the image for every lateral position of the
    # object, we need to zero-pad the image according to the lateral
    # detector size. We pad only the bottom of the image, putting its
    # center at the upper detector boundary.
    #pad2 = det_size*det_spacing - N/2
    pad2 = len(arrc[0]) - len(arrc)
    padset = np.pad(arrc, ((0,pad2), (0,0)), mode="constant")
    numsteps = int(np.ceil(N/shift_size))
    numangles = len(angles)
    lino = np.zeros((numsteps, numangles))
    
    
    import os
    import sys
    DIR = "."
    sys.path.append(os.path.realpath(DIR+"/../../../radontea/"))
    sys.path.append(os.path.realpath(DIR+"/../tomography-reconstruction"))
    sys.path.append(os.path.realpath(DIR+"/../FDTD"))
    import tool

    
    
    Nbound = int(np.floor(N/2))
    
    for i in range(numsteps):
        print(i)
        padset = np.roll(padset, shift_size, axis=0)
        # cut out a det_size slice
        curobj = padset#[Nbound:Nbound+det_size]
        for j in range(numangles):
            ang = angles[j]
            rotated = scipy.ndimage.rotate(curobj, ang/np.pi*180,
                        order=3, reshape=False, mode="constant", cval=0)
            #if i == 1:#int(numsteps/2):
            #    #rotated[centerid,:] = np.max(rotated)
            #    tool.arr2im(rotated, scale=True).save("./test/{:04d}.png".format(j))
            if det_size%2 != 0: #odd
                centerid = int(np.floor(len(rotated)/2)+1)
            else: #even
                centerid = int(len(rotated)/2)
                
            #lino[i,j] = np.sum(rotated[centerid, N+2*lS-1:])
            lino[i,j] = np.sum(rotated[centerid, :])
        
        #rotated[centerid,:] = np.max(rotated)    
        tool.arr2im(rotated, scale=True).save("./test/{:04d}.png".format(i))
            
        if callback is not None:
            callback(**cb_kwargs)
    if return_ang:
        return lino, angles
    else:
        return lino
