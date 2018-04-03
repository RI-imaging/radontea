import numpy as np


def volume_recon(func2d, sinogram=None, angles=None,
                 count=None, max_count=None, ncpus=None, **kwargs):
    """3D inverse with the Fourier slice theorem

    Computes the slice-wise 3D inverse of the radon transform using
    multiprocessing.


    Parameters
    ----------
    func2d: callable
        A method representing the 2D algorithm.
    sinogram: ndarray, shape (A,M,N)
        Three-dimensional sinogram of line recordings. The dimension `M`
        iterates through the slices. The rotation takes place through
        the second (y) axis.
    angles: (A,) ndarray
        Angular positions of the `sinogram` in radians equally
        distributed from zero to PI.
    count, max_count: multiprocessing.Value or `None`
        Can be used to monitor the progress of this function. The
        value of `max_count.value` is set initially and the value
        of `count.value` is incremented until it reaches the end
        of the algorithm (`max_count.value`).
    *kwargs: dict
        Additional keyword arguments to `func2d`.

    Returns
    -------
    out : ndarray
        The reconstructed image.

    """
    if sinogram.shape[0] != angles.shape[0]:
        msg = "First dimension of `sinogram` must match size of `angles`"
        raise ValueError(msg)

    if len(sinogram.shape) != 3:
        msg = "`sinogram` must have three dimensions."
        raise ValueError(msg)

    # TODO
    # - use multiprocessing
    # - define initial array

    results = []
    for ii in range(sinogram.shape[1]):
        resii = func2d(sinogram=sinogram[:, ii, :],
                       angles=angles,
                       count=count,
                       max_count=max_count,
                       **kwargs)
        results.append(resii)
    results = np.array(results)

    return results
