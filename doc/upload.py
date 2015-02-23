import os
import sys
import subprocess as sp

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# compile
sp.check_output([sys.executable, 'make.py'])

# checkout the gh-pages branch
sp.check_output(["git", 'checkout gh-pages'])

# copy built files
sp.check_output(["git", 'commit -a -m "automated "'])
