#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    _Back_3D.py

    Generates 3D variants of the 2D sinogram inversion methods of
    `radontea` using `multiprocessing`.
"""
# TODO
#
# Jobmanager:
# - Local Manager?
# - Example where a function calls a number of other functions with
#   only one Python call and returning a result?
#
from __future__ import division, print_function

import multiprocessing as mp
import numpy as np


from ._Back import *
from ._Back_iterative import *

from . import _Back
from . import _Back_iterative


__all__ = _Back.__all__ + _Back_iterative.__all__

modules_2d = [_Back, _Back_iterative]


def back_3d(sinogram=None, angles=None, method="backproject",
            jmc=None, jmm=None, **kwargs):
    u""" 3D inverse with the Fourier slice theorem

    Computes the slice-wise 3D inverse of the radon transform using
    multiprocessing.


    Parameters
    ----------
    sinogram : ndarray, shape (A,M,N)
        Three-dimensional sinogram of line recordings. The dimension `M`
        iterates through the slices. The rotation takes place through
        the second (y) axis.
    angles : (A,) ndarray
        Angular positions of the `sinogram` in radians equally
        distributed from zero to PI. Can also be given in `kwargs`.
    method : str
        The method to be used for sinogram inversion (e.g. "sart",
        "backproject", or "foutier_map"). `method` must be a method
        in :py:mod:`radontea`. This parameter is automatically
        overwritten with :py:func:`radontea.METHOD` unless you are
        explicitly calling `radontea._Back_3D.back_3d`.
    jmc, jmm : instance of :func:`multiprocessing.Value` or `None`
        The progress of this function can be monitored with the 
        :mod:`jobmanager` package. The current step `jmc.value` is
        incremented `jmm.value` times. `jmm.value` is set at the 
        beginning.
        This is not yet implemented!
    *kwargs : dict
        Keyword arguments to `radontea.METHOD`.

    Returns
    -------
    out : ndarray
        The reconstructed image.

    """
    assert sinogram.shape[0] == angles.shape[0], \
        "First dimension of `sinogram` must match size of `angles`"
    assert len(sinogram.shape) == 3, \
        "`sinogram` must have three dimensions."

    # Check if the method exists
    if isinstance(method, str):
        i = 0
        while i < len(modules_2d):
            try:
                func = getattr(modules_2d[i], method)
                break
            except:
                pass
            i += 1
        if i == len(modules_2d):
            raise ValueError("Unknown method: '{}'".format(method))

    # Write method to globals so we can wrap it
    _set_method(func.__name__)

    if angles is not None:
        kwargs["angles"] = angles

    (A, M, N) = sinogram.shape

    # How long will the algorithm run? - `jobmanager` counters.
    #kwargs["jmc"] = jmc
    # if func.__name__ == "backproject":
    #    if jmm is not None:
    #        jmm.value = M * (A+1)

    # arguments of the function
    func_args = func.__code__.co_varnames[:func.__code__.co_argcount]
    # default keyword arguments

    func_def = func.__defaults__[::-1]

    # build arguments reversely
    arglistsino = list()
    for m in range(M):
        arglist = list()
        kwargs["sinogram"] = sinogram[:, m, :]
        for i, a in enumerate(func_args[::-1]):
            # first set default
            if i < len(func_def):
                val = func_def[i]
            if a in kwargs:
                val = kwargs[a]
            arglist.append(val)
        arglistsino.append(arglist[::-1])

    # Map sinogram to cpus
    # Splice sinogram
    p = mp.Pool(mp.cpu_count())

    result = p.map_async(_wrapper_func, arglistsino).get()
    p.close()
    p.terminate()
    p.join()

    shape = (N, M, N)
    data = np.zeros(shape)
    for m in range(M):
        data[:, m, :] = result[m]

    del result

    return data


def _wrapper_func(args):
    # get the method
    for module in modules_2d:
        if hasattr(module, _used_method):
            func = getattr(module, _used_method)
            return func(*args)
    # no?
    raise AttributeError("Something went wrong.")


def _set_method(method):
    global _used_method
    _used_method = method


def _generate_3dfunc(method3d):
    if method3d.endswith("_2d") or method3d.endswith("_3d"):
        method = method3d[:-3]
    else:
        method = method3d
    # wraps around back_3d

    def _generated(*args, **kwargs):
        # translate args to kwargs
        func_args = back_3d.__code__.co_varnames[:back_3d.__code__.co_argcount]
        for i, arg in enumerate(args):
            kwargs[func_args[i]] = arg
        kwargs["method"] = method
        return back_3d(**kwargs)

    _generated.__doc__ = back_3d.__doc__.replace("METHOD", method)

    return _generated


_used_method = "none"

# overwrite globals function defined in 2d backprojection
# files.

# Todo:
# write function to get 2d functions and make this more clean.
glob = globals()
for method3d in __all__:
    glob[method3d] = _generate_3dfunc(method3d)
