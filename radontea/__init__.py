#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""
Introduction
============

There are several methods to compute the inverse **Radon** transform.
The module :py:mod:`radontea` implements some of them. I focussed on
code readability and thorough comments. The result is a collection of
algorithms that are suitable for **tea**\ching the basics of
computerized tomography.

For a quick overview, see :ref:`genindex`.


Methods
=======

"""

from ._Back import *
from . import _Back_2D as _2d
from . import _Back_3D as _3d
from ._Back_Fan import *
from ._Back_iterative import *
from ._Radon import *


__version__ = "0.1.5"
__author__ = "Paul Mueller"
__email__ = "paul.mueller@biotec.tu-dresden.de"
__license__ = "BSD (3-Clause)"
