#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""
There are several methods to compute the inverse **Radon** transform.
The module :py:mod:`radontea` implements some of them. I focussed on
code readability and thorough comments. The result is a collection of
algorithms that are suitable for **tea**\ching the basics of
computerized tomography.

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
