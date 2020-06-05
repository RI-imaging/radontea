#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists, dirname, realpath
from setuptools import setup, find_packages
import sys


author = u"Paul Müller"
authors = [author]
description = 'Collection of algorithms to compute the inverse Radon transform'
name = 'radontea'
year = "2014"

sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
from _version import version


setup(
    name=name,
    author=author,
    author_email="dev@craban.de",
    version=version,
    license="BSD (3 clause)",
    url='https://github.com/RI-imaging/radontea',
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    description=description,
    long_description=open('README.rst').read() if exists('README.rst') else '',
    install_requires=["numpy>=1.5.1",
                      "scipy>=1.4.0",  # Updated QHull in griddata
                      ],
    setup_requires=['pytest-runner'],
    tests_require=["pytest"],
    python_requires='>=3.5, <4',
    keywords=["tomography", "ct", "radon", "computerized tomography",
              "optical projection tomography"],
    classifiers= [
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization',
        'Intended Audience :: Science/Research'
                 ],
    platforms=['ALL'],
    )
