import os
import sys
import subprocess as sp

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# remove old html files
sp.check_output(["rm", '-r _build/html'])

# compile docs
sp.check_output(["make", 'html'])
