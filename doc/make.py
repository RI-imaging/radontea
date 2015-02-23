import os
import sys
import subprocess as sp

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sp.check_output(["make", 'html'])
