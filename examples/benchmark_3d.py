"""Volumetric data reconstruction benchmark

This simple example can be used to quantify the speed-up due to
multiprocessing on the hardware used. Note that
``multiprocessing.cpu_count`` does not return the number of
physical cores.
"""
from multiprocessing import cpu_count
import time

import numpy as np

import radontea as rt


A = 70    # number of angles
N = 128   # detector size x
M = 24    # detector size y (number of slices)

# generate random data
sino0 = np.random.random((A, N))     # for 2d example
sino = np.random.random((A, M, N))    # for 3d example
sino[:, 0, :] = sino0
angles = np.linspace(0, np.pi, A)     # for both

a = time.time()
data1 = rt.backproject_3d(sino, angles, ncpus=1)
print("time on 1 core:  {:.2f} s".format(time.time() - a))

a = time.time()
data2 = rt.backproject_3d(sino, angles, ncpus=cpu_count())
print("time on {} cores: {:.2f} s".format(cpu_count(), time.time() - a))

assert np.all(data1 == data2), "2D and 3D results don't match"
