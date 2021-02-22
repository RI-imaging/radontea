import pathlib

import numpy as np

import radontea
import sinogram


def test_2d_int():
    sino, angles = sinogram.create_test_sino(A=20, N=30)
    r = radontea.integrate(sino, angles)

    # np.savetxt('outfile.txt', np.array(r).flatten().view(float), fmt="%.8f")
    reffile = pathlib.Path(__file__).parent / "data" / "2d_int.txt"
    ref = np.loadtxt(str(reffile))
    assert np.allclose(np.array(r).flatten().view(float), ref)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
