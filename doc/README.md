radontea documentation
============================

Install [numpydoc](https://pypi.python.org/pypi/numpydoc):

    pip install numpydoc

To compile the documentation, run

    python setup.py build_sphinx


To upload the documentation to gh-pages, run

    python setup.py commit_doc
    
or

    cd doc
    python commit_gh-pages.py
