#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    _logo.py
    
    Functions that are used to create the logo of radontea.
"""
from __future__ import division

import numpy as np

if __name__ == "__main__":
    from _Radon import radon
else:
    from ._Radon import radon


def logo(x,y,N):
    """
        Vector representation of radonteach logo.
        
        
        Parameters
        ----------
        x,y : ndarray or float
            coordinates where to calculate the logo.
        N : float
            total size of the image (NxN)
            
            
        Returns
        -------
        v : ndarray
            inverted values of the logo at the coordinates (x,y)
            normalized to one.
    """
    z1 = np.exp(-((x-N/8)**2+(y+N/14)**2)/(N/8)**2)
    z2 = np.exp(-((x+N/10)**2+(y-N/10)**2)/(N/8)**2)

    z3 = np.exp(-((x-N/8)**2+(y+N/10)**2)/(N/12)**2)
    z4 = np.exp(-((x+N/10)**2+(y-N/14)**2)/(N/12)**2)

    z5 = np.exp(-((x)**2+(y)**2)/(N/4)**2)
    z = (z1+z2 -(z3+z4))
    z = (z-z.min())*z5
    z /= z.max()
    z[np.where(x**2+y**2>=(N/2)**2)] = 0
    
    return z


def get_original(N=64):
    """
        Returns a two-dimensional square image that is the image used
        to create the sinogram for the radontea logo.
    """
    x = np.linspace(-N/2, N/2, N, endpoint=False)
    X = x.reshape(1,-1)
    Y = x.reshape(-1,1)
    z = logo(X,Y,N)
    
    return np.array((z)*255, dtype=np.uint16)


def get_logo(N=64):
    """
        Returns a two-dimensional square logo of resolution NxN. This
        function discretizes the vector image representation of the
        radonteach logo.
    """
    a=(get_original(N=N))
    angles = np.linspace(0, np.pi*1.6, N) 
    sinogram = radon(a, angles)
    sinogram = (1-sinogram/sinogram.max())*255
    return sinogram.transpose()


def main():
    # Show the logo
    from matplotlib import pylab as plt
    from matplotlib import cm
    logo = get_logo(N=128)
    plt.imshow(logo, cmap=cm.gray)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    main()
