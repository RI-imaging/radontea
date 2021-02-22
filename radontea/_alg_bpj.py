import numpy as np
import scipy.interpolate as intp

from . import _threed, util


def backproject(sinogram: np.ndarray, angles: np.ndarray,
                filtering: str = "ramp", weight_angles: bool = True,
                padding: bool = True, padval: float = 0, count=None,
                max_count=None, verbose: int = 0) -> np.ndarray:
    r"""2D backprojection algorithm

    Computes the inverse of the Radon transform using filtered
    backprojection.


    Parameters
    ----------
    sinogram: ndarray, shape (A,N)
        Two-dimensional sinogram of line recordings.
    angles: (A,) ndarray
        Angular positions of the `sinogram` in radians.
    filtering: {'ramp', 'shepp-logan', 'cosine', 'hamming', \
                 'hann'}, optional
        Specifies the Fourier filter. Either of

          - "ramp": mathematically correct reconstruction
          - "shepp-logan"
          - "cosine"
          - "hamming"
          - "hann"

    weight_angles: bool
        If ``True``, weights each backpropagated projection with a factor
        proportional to the angular distance between the neighboring
        projections.

        .. math::
            \Delta \phi_0 \longmapsto \Delta \phi_j =
            \frac{\phi_{j+1} - \phi_{j-1}}{2}

        .. versionadded:: 0.1.9

    padding: bool, optional
        Pad the input data to the second next power of 2 before
        Fourier transforming. This reduces artifacts and speeds up
        the process for input image sizes that are not powers of 2.
    padval: float
        The value used for padding.
        If `padval` is `None`, then the edge values are used for
        padding (see documentation of `numpy.pad`).
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.
    verbose: int
        Increment to increase verbosity.

    Returns
    -------
    out: ndarray
        The reconstructed image.


    Examples
    --------
    >>> import numpy as np
    >>> from radontea import backproject
    >>> N = 5
    >>> A = 7
    >>> x = np.linspace(-N/2, N/2, N)
    >>> projection = np.exp(-x**2)
    >>> sinogram = np.tile(projection, A).reshape(A,N)
    >>> angles = np.linspace(0, np.pi, A, endpoint=False)
    >>> backproject(sinogram, angles)
    array([[ 0.03813283, -0.01972347, -0.02221885,  0.03822303,  0.01903376],
           [ 0.04033526, -0.01591647,  0.02262173,  0.04203002,  0.02123619],
           [ 0.02658797,  0.11375576,  0.41525594,  0.17170226,  0.0074889 ],
           [ 0.04008672,  0.11139209,  0.27650193,  0.16933859,  0.02098764],
           [ 0.02140445,  0.0334597 ,  0.07691547,  0.0914062 ,  0.00230537]])
    """
    # Perform weighting
    if weight_angles:
        weights = util.compute_angle_weights_1d(angles).reshape(-1, 1)
        sinogram = sinogram * weights
    else:
        sinogram = sinogram

    ln = sinogram.shape[1]
    la = angles.shape[0]
    assert sinogram.shape[0] == la

    if max_count is not None:
        with max_count.get_lock():
            max_count.value += la + 1

    # We perform padding before performing the Fourier transform.
    # This gets rid of artifacts due to false periodicity and also
    # speeds up Fourier transforms of the input image size is not
    # a power of 2.

    if padding:
        order = max(64., 2**np.ceil(np.log(ln * 2.1) / np.log(2)))
        pad = order - ln
    else:
        pad = 0
        order = ln

    padl = int(np.ceil(pad / 2))
    padr = int(pad - padl)

    if padval is None:
        if verbose > 0:
            print("......Padding with edge values.")
        sino = np.pad(sinogram, ((0, 0), (padl, padr)),
                      mode="edge")
        sino = np.roll(sino, -padl, 1)
    else:
        if verbose > 0:
            print("......Verifying padding value: {}".format(padval))
        sino = np.pad(sinogram, ((0, 0), (padl, padr)),
                      mode="linear_ramp",
                      end_values=(padval,))
        sino = np.roll(sino, -padl, 1)

    # These artifacts are for example bad contrast.
    kx = 2 * np.pi * np.abs(np.fft.fftfreq(int(order)))
    # Ask for the filter. Do not include zero (first element).
    if filtering == "ramp":
        pass
    elif filtering == "shepp-logan":
        kx[1:] = kx[1:] * np.sin(kx[1:]) / (kx[1:])
    elif filtering == "cosine":
        kx[1:] = kx[1:] * np.cos(kx[1:])
    elif filtering == "hamming":
        kx[1:] = kx[1:] * (0.54 + 0.46 * np.cos(kx[1:]))
    elif filtering == "hann":
        kx[1:] = kx[1:] * (1 + np.cos(kx[1:])) / 2
    elif filtering is None:
        kx[1:] = 2 * np.pi
    else:
        raise ValueError("Unknown filter: %s" % filter)

    if count is not None:
        with count.get_lock():
            count.value += 1
    # Resize f so we can multiply it with the sinogram.
    kx = kx.reshape(1, -1)
    projection = np.fft.fft(sino, axis=-1) * kx
    sino_filtered = np.real(np.fft.ifft(projection, axis=-1))
    # Resize filtered sinogram back to original size
    sino = sino_filtered[:, :ln]
    # Perform backprojection
    x = np.linspace(-ln / 2, ln / 2, ln, endpoint=False) + .5
    outarr = np.zeros((ln, ln), dtype=np.float64)
    # Meshgrid for output array
    xv, yv = np.meshgrid(x, x)

    for ii in np.arange(len(angles)):
        projinterp = intp.interp1d(x, sino[ii], fill_value=0.0,
                                   copy=False, bounds_error=False)
        # Call proj_interp with cos and sin of corresponding coordinate
        # and add it to the outarr.
        phi = angles[ii]
        # Smear projection onto 2d volume
        xp = xv * np.cos(phi) + yv * np.sin(phi)
        outarr += projinterp(xp)
        # Here a much slower version with the same result:
        # for j in x:
        #    for k in x:
        #        # shift points to match origin of coordinate system
        #        xv = j - ln/2
        #        yv = k - ln/2
        #        # perform a simple coordinate transform
        #        #
        #        xp = xv*np.cos(phi) + yv*np.sin(phi)
        #        # yp = -xv*np.sin(phi) + yv*np.cos(phi)
        #        projval = projinterp(xp)
        #        outarr[j][k] += projval
        if count is not None:
            with count.get_lock():
                count.value += 1

    # Normalize output (we assume that the projections are equidistant)
    # We measure angles in radians
    dphi = np.pi / len(angles)
    # The factor of 2π comes from the choice of the unitary angular
    # frequency Fourier transform.
    outarr *= dphi / (2 * np.pi)

    return outarr


def backproject_3d(sinogram: np.ndarray, angles: np.ndarray,
                   filtering: str = "ramp", weight_angles: bool = True,
                   padding: bool = True, padval: float = 0, count=None,
                   max_count=None, ncpus=None) -> np.ndarray:
    """Convenience wrapper for 3D backprojection reconstruction

    See :func:`backproject` for parameter definitions. The additional
    parameter `ncpus` sets the number of CPUs used.

    Returns
    -------
    out: ndarray, shape (N,M,N)
        The reconstructed volume.

        .. versionchanged:: 0.4.0

            Output indexing now follows the ODTbrain convention. For the
            the old behavior, use ``out.transpose(1, 0, 2)``.
    """
    return _threed.volume_recon(func2d=backproject,
                                sinogram=sinogram,
                                filtering=filtering,
                                weight_angles=weight_angles,
                                padding=padding,
                                padval=padval,
                                angles=angles,
                                count=count,
                                max_count=max_count,
                                ncpus=ncpus)
