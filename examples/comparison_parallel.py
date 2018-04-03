"""Comparison of parallel-beam reconstruction methods

This example illustrates the performance of the
different reconstruction techniques for a parallel-beam
geometry. The left column shows the reconstruction of
the original image and the right column shows the reconstruction
of the corresponding binary images. Note that the
SART process could be sped-up by computing an
initial guess with a non-iterative method and
setting it with the ``initial`` keyword argument.
"""
from matplotlib import pylab as plt
import numpy as np

import radontea
from radontea.logo import get_original

N = 55  # image size
A = 13  # number of sinogram angles
ITA = 10  # number of iterations a
ITB = 100  # number of iterations b

angles = np.linspace(0, np.pi, A)

im = get_original(N)
sino = radontea.radon_parallel(im, angles)
fbp = radontea.backproject(sino, angles)
fintp = radontea.fourier_map(sino, angles).real
sarta = radontea.sart(sino, angles, iterations=ITA)
sartb = radontea.sart(sino, angles, iterations=ITB)

im2 = (im >= (im.max() / 5)) * 255
sino2 = radontea.radon_parallel(im2, angles)
fbp2 = radontea.backproject(sino2, angles)
fintp2 = radontea.fourier_map(sino2, angles).real
sarta2 = radontea.sart(sino2, angles, iterations=ITA)
sartb2 = radontea.sart(sino2, angles, iterations=ITB)

plt.figure(figsize=(8, 22))
pltkw = {"vmin": -20,
         "vmax": 280}

plt.subplot(6, 2, 1, title="original image")
plt.imshow(im, **pltkw)
plt.axis('off')

plt.subplot(6, 2, 2, title="binary image")
plt.imshow(im2, **pltkw)
plt.axis('off')

plt.subplot(6, 2, 3, title="sinogram (from original)")
plt.imshow(sino)
plt.axis('off')

plt.subplot(6, 2, 4, title="sinogram (from binary)")
plt.imshow(sino2)
plt.axis('off')

plt.subplot(6, 2, 5, title="filtered backprojection")
plt.imshow(fbp, **pltkw)
plt.axis('off')

plt.subplot(6, 2, 6, title="filtered backprojection")
plt.imshow(fbp2, **pltkw)
plt.axis('off')

plt.subplot(6, 2, 7, title="Fourier interpolation")
plt.imshow(fintp, **pltkw)
plt.axis('off')

plt.subplot(6, 2, 8, title="Fourier interpolation")
plt.imshow(fintp2, **pltkw)
plt.axis('off')

plt.subplot(6, 2, 9, title="SART ({} iterations)".format(ITA))
plt.imshow(sarta, **pltkw)
plt.axis('off')

plt.subplot(6, 2, 10, title="SART ({} iterations)".format(ITA))
plt.imshow(sarta2, **pltkw)
plt.axis('off')

plt.subplot(6, 2, 11, title="SART ({} iterations)".format(ITB))
plt.imshow(sartb, **pltkw)
plt.axis('off')

plt.subplot(6, 2, 12, title="SART ({} iterations)".format(ITB))
plt.imshow(sartb2, **pltkw)
plt.axis('off')


plt.tight_layout()
plt.show()
