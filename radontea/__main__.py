#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Give the user a general idea of what this module does.
"""
import numpy as np

from ._Back import backproject, fourier_map
from ._Back_iterative import sart
from ._logo import get_original
from ._Radon import radon


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("wxagg")
    from matplotlib import pylab as plt
    from matplotlib import cm
        
    N=55
    It=7
    A=13

    # If available, use uilayer to display progress
    try:
        from uilayer import stdout
        steps = ((A+1)*2 + 4 + A*It+1)*2
        ui = stdout(show_warnings=False, steps=steps)
        callback = ui.Iterate
    except:
        ui = None
        callback = None

    angles = np.linspace(0,np.pi,A)
    im = get_original(N)
    
    sino = radon(im, angles, callback=callback)
    fbp = backproject(sino, angles, callback=callback)
    fintp = fourier_map(sino, angles, callback=callback)
    alg = sart(sino, angles, iterations=It, callback=callback)

    im2 = ( im >= (im.max()/5) ) * 255
    sino2 = radon(im2, angles, callback=callback)
    fbp2 = backproject(sino2, angles, callback=callback)
    fintp2 = fourier_map(sino2, angles, callback=callback)
    alg2 = sart(sino2, angles, iterations=It, callback=callback)

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
