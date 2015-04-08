#!/usr/bin/env python
# -*- coding: utf-8 -*-
# To create a distribution package for pip or easy-install:
# python setup.py sdist
from setuptools import setup, find_packages, Command
from os.path import join, dirname, realpath
import subprocess as sp
import sys
from warnings import warn


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



class PyDocGitHub(Command):
    """ Upload the docs to GitHub gh-pages branch
    """
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = sp.call([sys.executable, 'doc/commit_gh-pages.py'])
        raise SystemExit(errno)


class PyTest(Command):
    """ Perform pytests
    """
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = sp.call([sys.executable, 'tests/runtests.py'])
        raise SystemExit(errno)


if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        author_email="paul.mueller@biotec.tu-dresden.de",
        version=version,
        license="BSD (3 clause)",
        url='https://github.com/paulmueller/radontea',
        packages=[name],
        package_dir={name: name},
        description=description,
        long_description=open(join(dirname(__file__), 'README.txt')).read(),
        install_requires=[ "NumPy >= 1.5.1", "SciPy >= 0.8.0"],
        keywords=["tomography", "ct", "radon", "computerized tomography",
                  "optical projection tomography"],
        extras_require={
                        'doc': ['sphinx']
                       },
        classifiers= [
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Topic :: Scientific/Engineering :: Visualization',
            'Intended Audience :: Science/Research'
                     ],
        platforms=['ALL'],
        cmdclass = {'test': PyTest,
                    'commit_doc': PyDocGitHub,
                    },
        )

