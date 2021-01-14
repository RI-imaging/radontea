import numpy as np
import scipy.interpolate

from ._rdn_fan import get_fan_coords
from ._rdn_fan import get_det_coords, radon_fan  # noqa F401

from typing import Callable


def fan_rec(linogram: np.ndarray, lds: float, method: Callable, stepsize=1,
            det_spacing: float = 1, numang: int = None, count=None,
            max_count=None, **kwargs):
    """2D synthetic aperture reconstruction

    Computes the inverse of the fan-beam Radon transform using
    interpolation of the linogram and one of the inverse algorithms
    for tomography with the Fourier slice theorem.

    Parameters
    ----------
    linogram: 2d ndarray of shape (D, A)
        Input linogram from the synthetic aperture measurement.
    lds: float
        Distance in pixels between source and detector.
    method: callable
        Reconstruction method, e.g. `radontea.backproject`.
    numang: int
        Number of angles to be used for the sinogram. A higher number
        increases quality, but interpolation takes longer. By default
        numang = linogram.shape[1].
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.
    **kwargs: dict
        Keyword arguments for `method`.
    """
    sino, angles = lino2sino(linogram, lds, numang=numang, retang=True,
                             stepsize=stepsize, det_spacing=det_spacing,
                             count=count, max_count=max_count)
    return method(sino, angles, **kwargs)


def lino2sino(linogram: np.ndarray, lds: float, stepsize: float = 1,
              det_spacing: float = 1, numang: int = None,
              retang: bool = False,
              count=None, max_count=None) -> np.ndarray:
    """Convert linogram to sinogram for an equispaced detector.

    Parameters
    ----------
    linogram: real 2d ndarray of shape (D, A*)
        Linogram from synthetic aperture measurements.
    lds: float
        Distance from point source to detector in au.
    stepsize: float
        Translational increment of object in au (stepsize in D).
    det_spacing: float
        Distance between detector positions in au.
    numang: int
        Number of equispaced angles, defaults to linogram.shape[1]
    retang: bool
        Return the corresponding angles for the sinogram.
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.

    Returns
    -------
    sinogram: 2d ndarray of shape (D, A)
        The distortion-corrected sinogram.
        If retang is True, then the equispaced angles are returned as
        well.

    Notes
    -----
    This function can be used to convert a linogram obtained with
    fan-beam tomography to a sinogram, which then can be reconstructed
    with the backprojection or fourier mapping algorithms.
    """
    if not np.isreal(linogram):
        raise ValueError("`linogram` must be a real-valued array!")

    (D, det_size) = linogram.shape

    if numang is None:
        A = det_size  # (detector size determines # of angles)
    else:
        A = numang

    if max_count is not None:
        with max_count.get_lock():
            max_count.value += D + A

    # equispaced angles and corresponding lateral detector positions.
    angles, xang = get_fan_coords(det_size, det_spacing, lds, A)

    uorig = linogram
    lino = np.zeros((D, A))

    # go into the coordinate system of the original data
    xk = np.linspace(xang[0], xang[-1], det_size, endpoint=True)

    # begin equispacing
    for i in range(D):
        lino[i] = scipy.interpolate.spline(xk, uorig[i], xang)

        if count is not None:
            with count.get_lock():
                count.value += 1

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
        deltaD = np.tan(alpha) * lds / 2

        # Shearing:
        # At larger angles, the object seems bigger on the screen.
        xnew = xk / np.cos(alpha) + deltaD
        lino[:, i] = scipy.interpolate.spline(xk, lino[:, i], xnew)

        if count is not None:
            with count.get_lock():
                count.value += 1

    sino = np.transpose(lino)[:, ::-1]
    if retang:
        return sino, angles + np.pi / 2
    else:
        return sino
