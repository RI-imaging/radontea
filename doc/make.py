import os
import sys
import subprocess as sp

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sp.check_output(["make", 'html'])

os.chdir("..")

# checkout the gh-pages branch
sp.check_output(["git", 'checkout gh-pages'])

# copy 
sp.check_output(["git", 'commit -a -m "automated "'])
