#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" _Back_Fan.py

The inverse Radon transform with non-iterative techniques for
a fan-beam geometry.
"""
from __future__ import division

import numpy as np
import scipy

from ._Radon import get_angular_equispaced_coords

__all__= ["backproject_fan_translation",
          "lino2sino_equispaced_detector_2d"]


def backproject_fan_translation(linogram, angles, filtering="ramp",
                callback=None, cb_kwargs={}):
    pass




def lino2sino_equispaced_detector_2d(linogram, lDS, pxscale=None,
                                           stepsize=1, det_spacing=1,
                                           A=None):
    """ Convert linogram to sinogram for an equispaced detector.

    Parameters
    ----------
    linogram : real 2d ndarray of shape (D, A*)
        Linogram from synthetic aperture measurements.
    A : 1d ndarray
        Number of equispaced angles, defaults to linogram.shape[1]
    lDS : float
        Distance from point source to detector in pixels.
    pxscale : float
        One step in D in pixels


    Returns
    -------
    usin : 2d ndarray of shape (D, A)
        The distortion-corrected sinogram.        


    Notes
    -----
    This function can be used to convert a linogram obtained with
    fan-beam tomography to a sinogram, which then can be reconstructed
    with the backprojection or fourier mapping algorithms.
    """
    if not linogram is linogram.real:
        raise ValueError("`linogram` must be a real array!")
        
    (D, det_size) = linogram.shape
    
    if A is None:
        A = det_size #(detector size determines # of angles)
    
    
    if pxscale is None:
        pxscale = D/det_size

    # equispaced angles and corresponding lateral detector positions.
    angles, xang = get_angular_equispaced_coords(det_size, det_spacing, lDS, A)
    
    uorig = linogram
    lino = np.zeros((D, A))

    # go into the coordinate system of the original data
    xk = np.linspace(xang[0], xang[-1], det_size, endpoint=True)

    # begin equispacing
    for i in range(D):
        #uscaled[i] = scipy.interpolate.spline(xk, uorig[i], xang)
        lino[i] = scipy.interpolate.spline(xk, uorig[i], xang)

    # begin angular stretching
    for i in range(A):
        # parametrization of the time axis (spatial information)
        xk = np.linspace(-D*stepsize/2+.5,D*stepsize/2-.5, D, endpoint=True)
        alpha = angles[i]

        ## Centering:
        # The object moves 1px per view in D.
        # We need to translate this lateral movement to a shift that
        # depends on the current angle.
        # What is the distance b/w the center of the object (centered at
        # lDS/2) to the axis alpha = 0?
        deltaD = np.tan(alpha) * lDS/2
        #deltaD = np.tan(alpha) * D/2
        #xnew = xk + deltaD
        
        ## Sheering:
        # At larger angles, the object seems bigger on the screen.
        xnew = xk/np.cos(alpha)+deltaD
        lino[:,i] = scipy.interpolate.spline(xk, lino[:,i], xnew)
    
    return np.transpose(lino)[:,::-1]
