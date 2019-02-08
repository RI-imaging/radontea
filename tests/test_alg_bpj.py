import pathlib

import numpy as np

import radontea


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


def test_2d_backproject():
    sino, angles = create_test_sino(A=100, N=100)
    r = radontea.backproject(sino, angles, padding=True)

    # np.savetxt('outfile.txt', np.array(r).flatten().view(float), fmt="%.8f")
    reffile = pathlib.Path(__file__).parent / "data" / "2d_backproject.txt"
    ref = np.loadtxt(str(reffile))
    assert np.allclose(np.array(r).flatten().view(float), ref)


def test_3d_backproject():
    A = 100
    M = 30
    N = 100

    sino, angles = create_test_sino(A=100, N=100)
    sino3d = np.zeros((A, M, N), dtype=float)
    for ii in range(M):
        sino3d[:, ii, :] = sino
    r = radontea.backproject_3d(sino3d, angles, padding=True)
    reffile = pathlib.Path(__file__).parent / "data" / "2d_backproject.txt"
    ref = np.loadtxt(str(reffile))
    assert np.allclose(np.array(r[:, 0, :]).flatten().view(float), ref)
    assert np.all(r[:, 0, :] == r[:, 1, :])


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
