#!/usr/bin/env python
# To create a distribution package for pip or easy-install:
# python setup.py sdist
from setuptools import setup, find_packages
from os.path import join, dirname, realpath
from warnings import warn

import radontea

name='radontea'

setup(
    name=name,
    author=radontea.__author__,
    author_email=radontea.__email__,
    version=radontea.__version__,
    license=radontea.__license__,
    url='https://github.com/paulmueller/radontea',
    packages=[name],
    package_dir={name: name},
    description='Collection of algorithms to compute the inverse Radon transform',
    long_description=open(join(dirname(__file__), 'README.txt')).read(),
    install_requires=[ "NumPy >= 1.5.1", "SciPy >= 0.8.0"],
    keywords=["tomography", "ct", "radon"],
    extras_require={
                    'doc': ['sphinx']
                   },
    classifiers= [
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Topic :: Scientific/Engineering :: Visualization',
        'Intended Audience :: Science/Research'
                 ],
    platforms=['ALL']
    )

