==============
Code reference
==============

Parallel beam geometry
::::::::::::::::::::::
.. currentmodule:: radontea
.. autosummary:: 
    radon_parallel
    backproject
    fourier_map
    integrate
    art
    sart
    

Radon transform
---------------
.. autofunction:: radon_parallel

Non-iterative reconstruction
----------------------------
Computes the inverse Radon transform with non-iterative techniques.
The linear system of equations that describes the forward process can be
inverted with several algorithms, most notably the backprojection
algorithm :func:`radontea.backproject`. The reconstruction is based on
the Fourier slice theorem. A Fourier-based interpolation algorithm is
implemented in :func:`radontea.fourier_map`.

Backprojection
~~~~~~~~~~~~~~
.. autofunction:: backproject


Fourier mapping
~~~~~~~~~~~~~~~
.. autofunction:: fourier_map


Slow integration
~~~~~~~~~~~~~~~~
.. autofunction:: integrate


Iterative reconstruction
------------------------
Inversion of Radon-based tomography methods using iterative algorithms.
The convegence of these algorithms might be slow. The implementation
is not optimized.

ART
~~~
.. autofunction:: art

SART
~~~~
.. autofunction:: sart


Fan beam geometry
:::::::::::::::::
.. autosummary:: 
    fan.radon_fan
    fan.get_det_coords
    fan.get_fan_coords
    fan.lino2sino
    fan.fan_rec


Coordinate transforms
---------------------

.. autofunction:: radontea.fan.get_det_coords
.. autofunction:: radontea.fan.get_fan_coords
.. autofunction:: radontea.fan.radon_fan
.. autofunction:: radontea.fan.lino2sino


Reconstruction
--------------
.. autofunction:: radontea.fan.fan_rec


Volumetric reconstruction
:::::::::::::::::::::::::
For a slice-wise 3D reconstruction, radontea can use multiprocessing
to parallelize the reconstruction process.

Convenience functions
---------------------
.. autofunction:: backproject_3d

.. autofunction:: fourier_map_3d

General 3D reconstruction
-------------------------
.. autofunction:: volume_recon

