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


def radon(arr, angles, user_interface=None):
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
        user_interface : instance of `ttui.ui`, optional
            The user interface to which progress should be reported.
            The default is to output nothing.


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
    if user_interface is not None:
        pid = user_interface.progress_new(steps=len(angles),
                               task="projection.{}".format(os.getpid()))
    for i in np.arange(len(angles)):
        rotated = scipy.ndimage.rotate(arr, angles[i]/np.pi*180, order=3,
                  reshape=False, mode="constant", cval=0) #black corner
        # sum along some axis.
        outarr[i] = rotated.sum(axis=0)
        percent = i/len(angles)*100
        if user_interface is not None:
            user_interface.progress_iterate(pid)
    if user_interface is not None:
        user_interface.progress_finalize(pid)
    return outarr
