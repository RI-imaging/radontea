import os
import pathlib

import numpy as np

import pytest

import radontea
import sinogram


# See https://github.com/RI-imaging/radontea/issues/6
CI_FAILS = (os.environ.get("RUNNER_OS", "None") == "Linux")


@pytest.mark.xfail(CI_FAILS, reason="Unexplained issue #6")
def test_2d_backproject():
    sino, angles = sinogram.create_test_sino(A=100, N=100)
    r = radontea.backproject(sino, angles, padding=True)

    # np.savetxt('outfile.txt', np.array(r).flatten().view(float), fmt="%.8f")
    reffile = pathlib.Path(__file__).parent / "data" / "2d_backproject.txt"
    ref = np.loadtxt(str(reffile))
    assert np.allclose(np.array(r).flatten().view(float), ref)


@pytest.mark.xfail(CI_FAILS, reason="Unexplained issue #6")
def test_3d_backproject():
    A = 100
    M = 30
    N = 100

    sino, angles = sinogram.create_test_sino(A=100, N=100)
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
