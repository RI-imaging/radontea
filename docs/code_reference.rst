==============
Code reference
==============



Parallel beam geometry
::::::::::::::::::::::
.. autosummary:: 
    art
    backproject
    fourier_map
    radon
    sart
    sum


Radon transform
---------------
.. automodule:: radontea._Radon
.. currentmodule:: radontea
.. autofunction:: radon

Non-iterative reconstruction
----------------------------
.. currentmodule:: radontea
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


Sum
~~~
.. autofunction:: integrate


Iterative reconstruction
------------------------
.. currentmodule:: radontea
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
.. currentmodule:: radontea

.. autosummary:: 
    get_det_coords
    get_fan_coords
    lino2sino
    radon_fan_translation
    sa_interpolate


Coordinate transforms
---------------------

.. autofunction:: get_det_coords
.. autofunction:: get_fan_coords
.. autofunction:: lino2sino
.. autofunction:: radon_fan_translation


Non-iterative reconstruction
----------------------------
.. currentmodule:: radontea
The inverse Radon transform with non-iterative techniques for
a fan-beam geometry.

Interpolation
~~~~~~~~~~~~~
.. autofunction:: sa_interpolate

