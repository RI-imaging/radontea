#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" _Back_Fan.py

The inverse Radon transform with non-iterative techniques for
a fan-beam geometry.
"""
from __future__ import division

import numpy as np
import scipy

from ._Radon import get_fan_coords

__all__ = ["sa_interpolate",
           "lino2sino"]


def lino2sino(linogram, lDS, stepsize=1, det_spacing=1, numang=None,
              retang=False, jmc=None, jmm=None):
    """ Convert linogram to sinogram for an equispaced detector.

    Parameters
    ----------
    linogram : real 2d ndarray of shape (D, A*)
        Linogram from synthetic aperture measurements.
    lDS : float
        Distance from point source to detector in au.
    stepsize : float
        Translational increment of object in au (stepsize in D).
    det_spacing : float
        Distance between detector positions in au.
    numang : int
        Number of equispaced angles, defaults to linogram.shape[1]
    retang : bool
        Return the corresponding angles for the sinogram.
    jmc, jmm : instance of :func:`multiprocessing.Value` or `None`
        The progress of this function can be monitored with the 
        :mod:`jobmanager` package. The current step `jmc.value` is
        incremented `jmm.value` times. `jmm.value` is set at the 
        beginning.


    Returns
    -------
    sinogram : 2d ndarray of shape (D, A)
        The distortion-corrected sinogram.        
        If retang is True, then the equispaced angles are returned as
        well.


    Notes
    -----
    This function can be used to convert a linogram obtained with
    fan-beam tomography to a sinogram, which then can be reconstructed
    with the backprojection or fourier mapping algorithms.


    See Also
    --------
    radon_fan_translation
        The forward process.
    sa_interpolate
        Backprojection that uses this function.
    """
    if not linogram is linogram.real:
        raise ValueError("`linogram` must be a real array!")

    (D, det_size) = linogram.shape

    if numang is None:
        A = det_size  # (detector size determines # of angles)
    else:
        A = numang

    if jmm is not None:
        jmm.value = D + A

    # equispaced angles and corresponding lateral detector positions.
    angles, xang = get_fan_coords(det_size, det_spacing, lDS, A)

    uorig = linogram
    lino = np.zeros((D, A))

    # go into the coordinate system of the original data
    xk = np.linspace(xang[0], xang[-1], det_size, endpoint=True)

    # begin equispacing
    for i in range(D):
        #uscaled[i] = scipy.interpolate.spline(xk, uorig[i], xang)
        lino[i] = scipy.interpolate.spline(xk, uorig[i], xang)

        if jmc is not None:
            jmc.value += 1

    # begin angular stretching
    for i in range(A):
        # parametrization of the time axis (spatial information)
        xk = np.linspace(-D * stepsize / 2 + .5,
                         D * stepsize / 2 - .5, D, endpoint=True)
        alpha = angles[i]

        # Centering:
        # The object moves 1px per view in D.
        # We need to translate this lateral movement to a shift that
        # depends on the current angle.
        # What is the distance b/w the center of the object (centered at
        # lDS/2) to the axis alpha = 0?
        deltaD = np.tan(alpha) * lDS / 2
        #deltaD = np.tan(alpha) * D/2
        #xnew = xk + deltaD

        # Sheering:
        # At larger angles, the object seems bigger on the screen.
        xnew = xk / np.cos(alpha) + deltaD
        lino[:, i] = scipy.interpolate.spline(xk, lino[:, i], xnew)

        if jmc is not None:
            jmc.value += 1

    sino = np.transpose(lino)[:, ::-1]
    if retang:
        return sino, angles + np.pi / 2
    else:
        return sino


def sa_interpolate(linogram, lDS, method, stepsize=1, det_spacing=1,
                   numang=None, jmm=None, jmc=None, **kwargs):
    """ 2D synthetic aperture reconstruction

    Computes the inverse of the fan-beam radon transform using inter-
    polation of the linogram and one of the inverse algorithms for
    tomography with the Fourier slice theorem.

    Parameters
    ----------
    linogram : 2d ndarray of shape (D, A)
        Input linogram from the synthetic aprture measurement.
    lDS : float
        Distance in pixels between source and detector.
    method : callable
        Reconstruction method, e.g. `radontea.backproject`.
    numang : int
        Number of angles to be used for the sinogram. A higher number
        increases quality, but interpolation takes longer. By default
        numang = linogram.shape[1].
    jmc, jmm : instance of :func:`multiprocessing.Value` or `None`
        The progress of this function can be monitored with the 
        :mod:`jobmanager` package. The current step `jmc.value` is
        incremented `jmm.value` times. `jmm.value` is set at the 
        beginning.
    **kwargs : dict
        Keyword arguments for `method`.


    See Also
    --------
    radon_fan_translation
        The forward process.
    lino2sino
        Linogram to sinogram conversion.
    """
    sino, angles = lino2sino(linogram, lDS, numang=numang, retang=True,
                             stepsize=stepsize, det_spacing=det_spacing,
                             jmm=jmm, jmc=jmc)
    #kwargs["jmm"] = jmm
    #kwargs["jmc"] = jmc
    return method(sino, angles, **kwargs)
