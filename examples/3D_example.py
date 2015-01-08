#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Example of handling 3D data.

Using the `*_3d` methods only speeds up calculations if your processor
has multiple cores.
"""
from __future__ import print_function, division
from multiprocessing import cpu_count
import numpy as np
from os.path import split, dirname, abspath
import time
import sys

sys.path = [split(dirname(abspath(__file__)))[0]] + sys.path

import radontea as rt

A = 70    # number of angles
N = 128   # detector size x
M = 24    # detector size y (number of slices)

# generate random data
sino0 = np.random.random((A,N))     # for 2d example
sino = np.random.random((A,M,N))    # for 3d example
sino[:,0,:] = sino0
angles = np.linspace(0,np.pi,A)     # for both


a = time.time()
data0 = rt.backproject(sino0, angles)
print("time on 1 core: {} s".format((time.time() - a)*M))


a = time.time()
data = rt.backproject_3d(sino, angles)
print("time on {} cores: {} s".format(cpu_count(), time.time() - a))


assert np.sum(data0==data[:,0,:]) == N**2, "2D and 3D results don't match"
