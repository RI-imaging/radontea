#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""

There are several methods to compute the inverse **Radon** transform.
The module :py:mod:`radontea` implements some of them. I focussed on
code readability and thorough comments. The result is a collection of
algorithms that are suitable for **tea**\ching the basics of
computerized tomography.


Recommended literature 
----------------------
- Aninash C. Kak and Malcom Slaney. *Principles of Computerized 
  Tomographic Imaging*. Ed. by Robert E. O’Malley. SIAM, 2001, p. 327.
  ISBM: 089871494X.

- Johann Radon. *Über die Bestimmung von Funktionen durch ihre
  Integralwerte längs gewisser Mannigfaltigkeiten*. Tech. rep. Leipzig:
  Berichte über die Verhandlungen der Königlich-Sächsischen Gesellschaft
  der Wissenschaften zu Leipzig, 1917, pp. 262–277.

- R A Crowther, D J DeRosier, and A Klug. *The Reconstruction of a
  Three-Dimensional Structure from Projections and its Application to
  Electron Microscopy*. In: Proceedings of the Royal Society of London.
  A. Mathematical and Physical Sciences 317.1530 (1970), pp. 319–340.
  doi: `10.1098/rspa.1970.0119 
  <http://dx.doi.org/10.1098/rspa.1970.0119>`_.

Obtaining radontea 
------------------
If you have Python and :py:mod:`numpy` installed, simply run

    pip install radontea

The source code of radontea is available at
https://github.com/paulmueller/radontea.


Citing radontea
---------------
The radontea package should be cited like this (replace "x.x.x"
with the actual version of radontea that you used and "DD Month YYYY"
with a matching date).

.. topic:: cite

    Paul Müller (2013) *Python algorithms for the inversion of the
    Radon transform* (Version x.x.x)
    [Computer program].
    Available at https://pypi.python.org/pypi/radontea/
    (Accessed DD Month YYYY)


You can find out what version you are using by typing
(in a Python console):


    >>> import radontea
    >>> radontea.__version__
    '0.1.4'

"""

from ._Back_meta import *
#from . import _Back as _2d
#from . import _Back_3D as _3d
from ._Back_Fan import *
#from ._Back_iterative import *
from ._Radon import *


from ._version import version as __version__
__author__ = u"Paul Müller"
__license__ = "BSD (3-Clause)"
