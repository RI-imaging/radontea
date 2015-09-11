#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests backprojection algorithm
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import join
import sys
import time

from os.path import abspath, dirname, split

# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import radontea
import radontea._Back


def create_test_sino(A, N):
    """
    Creates test sinogram.
    """
    resar = np.zeros((A,N), dtype=float)
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
    resar = 2*dev*resar/resar.max()
    return resar, angles


def test_2d_backproject():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    
    sino, angles = create_test_sino(A=100, N=100)
    r = radontea.backproject(sino, angles, padding=True)
    r2 = radontea._Back.backproject(sino, angles, padding=True)
    
    #np.savetxt('outfile.txt', np.array(r).flatten().view(float), fmt="%.8f")
    assert np.allclose(r, r2)
    assert np.allclose(np.array(r).flatten().view(float), results[myname])



# Get results
results = dict()
datadir = join(DIR, "data")
for f in os.listdir(datadir):
    #np.savetxt('outfile.txt', np.array(r).flatten().view(float), fmt="%.10f")
    #np.savetxt('outfile.txt', np.array(r).flatten().view(float))
    glob = globals()
    if f.endswith(".txt") and f[:-4] in list(glob.keys()):
        results[f[:-4]] = np.loadtxt(join(datadir, f))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
