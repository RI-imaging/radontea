import numpy as np
import scipy.ndimage


def radon_parallel(arr: np.ndarray, angles: np.ndarray,
                   count=None, max_count=None,) -> np.ndarray:
    """Compute the Radon transform (sinogram) of a circular image.

    The :mod:`scipy` Radon transform performs this operation on the
    entire image, whereas this implementation requires an input
    image that has gray-scale values of 0 outside of a circle
    with diameter equal to the image size.


    Parameters
    ----------
    arr: ndarray, shape (N,N)
        the input image.
    angles: ndarray, length A
        angles or projections in radians
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of the algorithm.
        Initially, the value of `max_count.value` is incremented
        by the total number of steps. At each step, the value
        of `count.value` is incremented.


    Returns
    -------
    outarr: ndarray of floats, shape (A, N)
        Sinogram of the input image. The i'th row contains the
        projection data of the i'th angle.


    See Also
    --------
    scipy.ndimage.interpolation.rotate :
        The interpolator used to rotate the image.
    """
    # This function also works with single angles
    angles = np.atleast_1d(angles)
    A = angles.shape[0]
    N = len(arr)
    # The radon function from skimage.transform does not allow
    # to reshape the image (one could cut it of course).
    # outarray: x-axis: projection
    #           y-axis:
    outarr = np.zeros((A, N))

    # progress monitoring
    if max_count is not None:
        with max_count.get_lock():
            max_count.value += A + 1
    if count is not None:
        with count.get_lock():
            count.value += 1

    for i in np.arange(A):
        rotated = scipy.ndimage.rotate(arr,
                                       angles[i] / np.pi * 180,
                                       order=3,
                                       reshape=False,
                                       mode="constant",
                                       cval=0)
        # sum along the axis.
        outarr[i] = rotated.sum(axis=0)
        if count is not None:
            with count.get_lock():
                count.value += 1
    return outarr
