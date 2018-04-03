import numpy as np

import radontea


def test_radon_simple():
    x = np.zeros((10, 10))
    x[2:5, 3:6] = 1
    sino = radontea.radon_parallel(x, angles=[0, np.pi/2, np.pi, np.pi*3/2])
    assert np.allclose(sino[0], [0, 0, 0, 3, 3, 3, 0, 0, 0, 0])
    assert np.allclose(sino[1], [0, 0, 3, 3, 3, 0, 0, 0, 0, 0])
    assert np.allclose(sino[2], [0, 0, 0, 0, 3, 3, 3, 0, 0, 0])
    assert np.allclose(sino[3], [0, 0, 0, 0, 0, 3, 3, 3, 0, 0])


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
