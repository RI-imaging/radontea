#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    There are several methods to compute the inverse *Radon* transform.
    The module *radontea* implements some of them. I focussed on code
    readability and thorough comments. The result is a collection of
    algorithms that are suitable for *tea*\ching the basics of
    computerized tomography.
"""

from ._Back import *
from ._Back_iterative import *
from ._Radon import *

__version__ = "0.1.0"
__author__ = "Paul Mueller"
__email__ = "paul.mueller@biotec.tu-dresden.de"
__license__ = "OpenBSD"
