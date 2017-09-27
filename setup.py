#!/usr/bin/env python
# -*- coding: utf-8 -*-
# To create a distribution package for pip or easy-install:
# python setup.py sdist
from setuptools import setup
from os.path import exists, dirname, realpath
import sys


author = u"Paul Müller"
authors = [author]
description = 'Collection of algorithms to compute the inverse Radon transform'
name = 'radontea'
year = "2014"

sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version
except:
    version = "unknown"


if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        author_email="dev@craban.de",
        version=version,
        license="BSD (3 clause)",
        url='https://github.com/RI-imaging/radontea',
        packages=[name],
        package_dir={name: name},
        description=description,
        long_description=open('README.rst').read() if exists('README.rst') else '',
        install_requires=[ "NumPy >= 1.5.1", "SciPy >= 0.8.0"],
        keywords=["tomography", "ct", "radon", "computerized tomography",
                  "optical projection tomography"],
        setup_requires=['pytest-runner'],
        tests_require=["pytest"],
        classifiers= [
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Topic :: Scientific/Engineering :: Visualization',
            'Intended Audience :: Science/Research'
                     ],
        platforms=['ALL'],
        )

