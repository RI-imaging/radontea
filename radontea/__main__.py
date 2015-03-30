#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Give the user a general idea of what this module does.
"""
import numpy as np

from . import _Back as back
from . import _Back_iterative as back_it
from . import _Radon as radon
from ._logo import get_original

# If available, use jobmanager to display progress
try:
    import jobmanager as jm
    for module in [back, back_it, radon]:
        jm.decorators.decorate_module_ProgressBar(
            module, override_count=True, interval=.1, verbose=1)
except:
    pass


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("wxagg")
    from matplotlib import pylab as plt

    N = 55
    It = 7
    A = 13

    angles = np.linspace(0, np.pi, A)
    im = get_original(N)

    callback = None

    sino = radon.radon(im, angles)

    fbp = back.backproject(sino, angles)

    fintp = back.fourier_map(sino, angles)

    alg = back_it.sart(sino, angles, iterations=It)

    im2 = (im >= (im.max() / 5)) * 255
    sino2 = radon.radon(im2, angles)
    fbp2 = back.backproject(sino2, angles)
    fintp2 = back.fourier_map(sino2, angles)
    alg2 = back_it.sart(sino2, angles, iterations=It)

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 5, 1)
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('uint16 image')

    plt.subplot(2, 5, 2)
    plt.imshow(sino, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('sinogram ({} proj.)'.format(A))

    plt.subplot(2, 5, 3)
    plt.imshow(fbp, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('backprojection')

    plt.subplot(2, 5, 4)
    plt.imshow(fintp.real, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('Fourier interpolation')

    plt.subplot(2, 5, 5)
    plt.imshow(alg, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('SART ({} iterations)'.format(It))

    plt.subplot(2, 5, 6)
    plt.imshow(im2, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('binary image')

    plt.subplot(2, 5, 7)
    plt.imshow(sino2, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('sinogram ({} proj.)'.format(A))

    plt.subplot(2, 5, 8)
    plt.imshow(fbp2, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('backprojection')

    plt.subplot(2, 5, 9)
    plt.imshow(fintp2.real, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('Fourier interpolation')

    plt.subplot(2, 5, 10)
    plt.imshow(alg2, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title('SART ({} iterations)'.format(It))

    plt.show()
