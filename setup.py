#!/usr/bin/env python
# To create a distribution package for pip or easy-install:
# python setup.py sdist
from setuptools import setup, find_packages
from os.path import join, dirname, realpath
from warnings import warn

import radontea

setup(
    name='radontea',
    author='Paul Mueller',
    author_email='paul.mueller@biotec.tu-dresden.de',
    url='https://github.com/paulmueller/radontea',
    version=radontea.__version__,
    packages=['radontea'],
    package_dir={'radontea': 'radontea'},
    license="OpenBSD",
    description='Collection of algorithms to compute the inverse Radon transform',
    long_description=open(join(dirname(__file__), 'README.txt')).read(),
    install_requires=[ "NumPy >= 1.5.1",
                       "SciPy >= 0.8.0"  ]
    )

