radontea reference
==================
.. toctree::
   :maxdepth: 2

General
:::::::

.. automodule:: radontea

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
.. automodule:: radontea._Back
.. currentmodule:: radontea

Backprojection
~~~~~~~~~~~~~~
.. autofunction:: backproject


Fourier mapping
~~~~~~~~~~~~~~~
.. autofunction:: fourier_map


Sum
~~~
.. autofunction:: sum


Iterative reconstruction
------------------------
.. automodule:: radontea._Back_iterative
.. currentmodule:: radontea

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
.. automodule:: radontea._Back_Fan
.. currentmodule:: radontea

Interpolation
~~~~~~~~~~~~~
.. autofunction:: sa_interpolate

