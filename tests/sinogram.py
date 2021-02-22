"""Creates a sinogram for testing purposes"""
import numpy as np


def create_test_sino(A, N):
    """
    Creates test sinogram.
    """
    resar = np.zeros((A, N), dtype=float)
    angles = np.linspace(0, np.pi, A, endpoint=False)
    x = np.linspace(-N/2, N/2, N, endpoint=True)
    dev = np.sqrt(N/2)
    off = N/7
    for ii in range(A):
        # Gaussian distribution sinogram
        x0 = np.cos(angles[ii])*off
        y = np.exp(-(x-x0)**2/dev**2)
        resar[ii] = y

    # some normalization, so that max of Radon-inversion
    # is about 1.
    resar = 2 * dev * resar / resar.max()
    return resar, angles
