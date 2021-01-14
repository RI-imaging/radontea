import warnings

import numpy as np
import scipy.interpolate as intp

from . import _threed


def fourier_map(sinogram: np.ndarray, angles: np.ndarray,
                intp_method: str = "cubic",
                count=None, max_count=None) -> np.ndarray:
    """2D Fourier mapping with the Fourier slice theorem

    Computes the inverse of the Radon transform using Fourier
    interpolation.
    Warning: This is the naive reconstruction that assumes that
    the image is rotated through the upper left pixel nearest to
    the actual center of the image. We do not have this problem for
    odd images, only for even images.


    Parameters
    ----------
    sinogram: (A,N) ndarray
        Two-dimensional sinogram of line recordings.
    angles: (A,) ndarray
        Angular positions of the `sinogram` in radians equally
        distributed from zero to PI.
    intp_method: {'cubic', 'nearest', 'linear'}, optional
        Method of interpolation. For more information see
        `scipy.interpolate.griddata`. One of

          - "nearest": instead of interpolating, use the points closest
            to the input data
          - "linear": bilinear interpolation between data points
          - "cubic": interpolate using a two-dimensional poolynimial
            surface

    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.


    Returns
    -------
    out: ndarray
        The reconstructed image.


    See Also
    --------
    scipy.interpolate.griddata: interpolation method used
    """
    if max_count is not None:
        with max_count.get_lock():
            max_count.value += 4

    if len(sinogram[0]) % 2 == 0:
        warnings.warn("Fourier interpolation with slices that have" +
                      " even dimensions leads to image distortions!")
    # projections
    p_x = sinogram

    # Fourier transform of the projections
    # The sinogram data is shifted in Fourier space
    P_fx = np.fft.fft(np.fft.ifftshift(p_x, axes=-1), axis=-1)

    if count is not None:
        with count.get_lock():
            count.value += 1

    # This paragraph could be useful for future version if the reconstruction
    # grid is to be changed. They directly affect *P_fx*.
    # if False:
    #    # Resize everyting
    #    factor = 10
    #    P_fx = np.zeros((len(sinogram), len(sinogram[0])*factor),
    #                    dtype=np.complex128)
    #    newp = np.zeros((len(sinogram), len(sinogram[0])*factor),
    #                    dtype=np.complex128)
    #    for i in range(len(sinogram)):
    #        x = np.linspace(0, len(sinogram[0]), len(sinogram[0]))
    #        dint = intp.interp1d(x, sinogram[i])
    #        xn = np.linspace(0, len(sinogram[0]), len(sinogram[0])*factor)
    #        datan = dint(xn)
    #        datfft = np.fft.fft(datan)
    #        newp[i] = datan
    #        # shift datfft
    #        P_fx[i] = np.roll(1*datfft,0)
    #    p_x = newp
    # if False:
    #    # Resize the input image
    #    P_fx = np.zeros(p_x.shape, dtype=np.complex128)
    #    for i in range(len(sinogram)):
    #        factor = 2
    #        x = np.linspace(0, len(sinogram[0]), len(sinogram[0]))
    #        dint = intp.interp1d(x, sinogram[i])#, kind="nearest")
    #        xn = np.linspace(0, len(sinogram[0]), len(sinogram[0])*factor)
    #        datan = dint(xn)
    #        datfft = np.fft.fft(datan)
    #        fftint = intp.interp1d(xn, datfft)
    #        start = (len(xn) - len(x))/2
    #        end = (len(xn) + len(x))/2
    #        fidata = fftint(x)
    #        datfft = np.fft.fftshift(datfft)
    #        datfft = datfft[start:end]
    #        datfft = np.fft.ifftshift(datfft)
    #        P_fx[i] = 1*datfft

    # angles need to be normalized to 2pi
    # if angles star with 0, then the image is falsly rotated
    # unfortunately we still have ashift in the data.
    ang = (angles.reshape(-1, 1))

    # compute frequency coordinates fx
    fx = np.fft.fftfreq(len(p_x[0]))

    fx = fx.reshape(1, -1)
    # fy is zero
    fxl = (fx) * np.cos(ang)
    fyl = (fx) * np.sin(ang)
    # now fxl, fyl, and P_fx have same shape

    # DEBUG: plot coordinates of positions of projections in fourier domain
    #   from matplotlib import pylab as plt
    #   plt.figure()
    #   for i in range(len(fxl)):
    #      plt.plot(fxl[i],fyl[i],"x")
    #   plt.axes().set_aspect('equal')
    #   plt.show()

    # flatten everything for interpolation
    Xf = fxl.flatten()
    Yf = fyl.flatten()
    Zf = P_fx.flatten()
    # rintp defines the interpolation grid
    rintp = np.fft.fftshift(fx.reshape(-1))

    if count is not None:
        with count.get_lock():
            count.value += 1

    # The code block yields the same result as griddata (below)
    # interpolation coordinates
    #   Rf = np.zeros((len(Xf),2))
    #   Rf[:,0] = 1*Xf
    #   Rf[:,1] = 1*Yf
    # reconstruction coordinates
    #   Nintp = len(rintp)
    #   Xn, Yn = np.meshgrid(rintp,rintp)
    #   Rn = np.zeros((Nintp**2, 2))
    #   Rn[:,0] = Xn.flatten()
    #   Rn[:,1] = Yn.flatten()
    #
    #    if intp_method.lower() == "bilinear":
    #       Fr = intp.LinearNDInterpolator(Rf,Zf.real)
    #       Fi = intp.LinearNDInterpolator(Rf,Zf.imag)
    #    elif intp_method.lower() == "nearest":
    #       Fr = intp.NearestNDInterpolator(Rf,Zf.real)
    #       Fi = intp.NearestNDInterpolator(Rf,Zf.imag)
    #    else:
    #       raise NotImplementedError("Unknown interpolation type: {}".format(
    #                                  intp_method.lower()))
    #   Fcomp = (Fr(Rn) + 1j*Fi(Rn)).reshape(Nintp,Nintp)

    # The actual interpolation
    Fcomp = intp.griddata((Xf, Yf), Zf, (rintp[None, :], rintp[:, None]),
                          method=intp_method)

    if count is not None:
        with count.get_lock():
            count.value += 1

    # remove nans
    Fcomp[np.where(np.isnan(Fcomp))] = 0

    f = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fcomp)))

    if count is not None:
        with count.get_lock():
            count.value += 1

    return f.real


def fourier_map_3d(sinogram: np.ndarray, angles: np.ndarray,
                   intp_method: str = "cubic",
                   count=None, max_count=None, ncpus=None) -> np.ndarray:
    """Convenience wrapper for 3D Fourier mapping reconstruction

    See :func:`fourier_map` for parameter definitions. The additional
    parameter `ncpus` sets the number of CPUs used.

    Returns
    -------
    out: ndarray, shape (N,M,N)
        The reconstructed volume.

        .. versionchanged:: 0.4.0

            Output indexing now follows the ODTbrain convention. For the
            the old behavior, use ``out.transpose(1, 0, 2)``.
    """
    return _threed.volume_recon(func2d=fourier_map,
                                sinogram=sinogram,
                                angles=angles,
                                intp_method=intp_method,
                                count=count,
                                max_count=max_count,
                                ncpus=ncpus)
