import multiprocessing as mp

import numpy as np

import radontea


def test_simple():
    """Simple test for backproject"""
    count = mp.Value("i")
    max_count = mp.Value("i")
    sino = np.random.random((10, 30))
    radontea.backproject(sinogram=sino,
                         angles=np.arange(len(sino)),
                         count=count,
                         max_count=max_count)
    assert count.value == 11
    assert count.value == max_count.value


def test_max_count_all_parralel_2d():
    """Test that count always reaches max_count"""
    sino = np.random.random((10, 31))
    angles = np.arange(len(sino))
    for name in ["backproject",
                 "fourier_map",
                 "integrate",
                 "art",
                 "sart",
                 ]:

        method = getattr(radontea, name)
        count = mp.Value("i")
        max_count = mp.Value("i")
        method(sinogram=sino,
               angles=angles,
               count=count,
               max_count=max_count)
        assert max_count.value == count.value


def test_bpj_3d_convenience():
    """test that max_count is incremented correctly in 3D methods"""
    sino = np.random.random((10, 4, 31))
    angles = np.arange(len(sino))
    count = mp.Value("i")
    max_count = mp.Value("i")
    radontea.backproject_3d(sinogram=sino,
                            angles=angles,
                            count=count,
                            max_count=max_count)
    assert max_count.value == 44  # and not 11
    assert max_count.value == count.value


def test_fmp_3d_convenience():
    """test that max_count is incremented correctly in 3D methods"""
    sino = np.random.random((10, 4, 31))
    angles = np.arange(len(sino))
    count = mp.Value("i")
    max_count = mp.Value("i")
    radontea.fourier_map_3d(sinogram=sino,
                            angles=angles,
                            count=count,
                            max_count=max_count)
    assert max_count.value == 16  # and not 4
    assert max_count.value == count.value


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
