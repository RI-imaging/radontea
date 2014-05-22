#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Give the user a general idea of what this module does.
"""
import numpy as np

from ._Back import backproject, fourier_interp
from ._Back_iterative import sart
from ._logo import get_original
from ._Radon import radon


if __name__ == "__main__":
    from matplotlib import pylab as plt
    from matplotlib import cm
    try:
        from ttui import stdout
        ui = stdout()
    except:
        ui = None
        
    N=55
    It=7
    A=13
    angles = np.linspace(0,np.pi,A)
    im = get_original(N)
    
    sino = radon(im, angles, user_interface=ui)
    fbp = backproject(sino, angles, user_interface=ui)
    fintp = fourier_interp(sino, angles, user_interface=ui)
    alg = sart(sino, angles, iterations=It, user_interface=ui)

    im2 = ( im >= (im.max()/5) ) * 255
    sino2 = radon(im2, angles, user_interface=ui)
    fbp2 = backproject(sino2, angles, user_interface=ui)
    fintp2 = fourier_interp(sino2, angles, user_interface=ui)
    alg2 = sart(sino2, angles, iterations=It, user_interface=ui)

    plt.figure(figsize=(15,8))
    
    plt.subplot(2,5,1)
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('uint16 image')

    plt.subplot(2,5,2)
    plt.imshow(sino, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('sinogram ({} proj.)'.format(A))

    plt.subplot(2,5,3)
    plt.imshow(fbp, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('backprojection')
    
    plt.subplot(2,5,4)
    plt.imshow(fintp.real, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('Fourier interpolation')


    plt.subplot(2,5,5)
    plt.imshow(alg, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('SART ({} iterations)'.format(It))

    plt.subplot(2,5,6)
    plt.imshow(im2, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('binary image')

    plt.subplot(2,5,7)
    plt.imshow(sino2, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('sinogram ({} proj.)'.format(A))
    
    plt.subplot(2,5,8)
    plt.imshow(fbp2, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('backprojection')
    
    plt.subplot(2,5,9)
    plt.imshow(fintp2.real, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('Fourier interpolation')

    plt.subplot(2,5,10)
    plt.imshow(alg2, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('SART ({} iterations)'.format(It))
    
    plt.show()
