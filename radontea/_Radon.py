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


__all__ = ["radon"]


def radon(arr, angles, trigger=None, **kwargs):
    """ 
        Compute the Radon transform (sinogram) of a circular image.
        
        The scipy Radon transform performs this operation on the 
        entire image, whereas this implementation requires an input
        image that has gray-scale values of 0 outside of a circle
        with diameter equal to image size.


        Parameters
        ----------
        arr : ndarray, shape (N,N)
            the input image.
        angles : ndarray, length A
            angles or projections in radians
        trigger : callable, optional
            If set, the function `trigger` is called on a regular basis
            throughout this algorithm.
            Number of function calls: A+1
        **kwargs : dict, optional
            Keyword arguments for trigger (e.g. "pid" of process).


        Returns
        -------
        outarr : ndarray of floats, shape (A,N)
            sinogram of the input image. The i'th row contains the
            projection data of i'th angle.


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
    if trigger is not None:
        trigger(**kwargs)
    for i in np.arange(len(angles)):
        rotated = scipy.ndimage.rotate(arr, angles[i]/np.pi*180, order=3,
                  reshape=False, mode="constant", cval=0) #black corner
        # sum along some axis.
        outarr[i] = rotated.sum(axis=0)
        percent = i/len(angles)*100
        if trigger is not None:
            trigger(**kwargs)

    return outarr
