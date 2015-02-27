import os
import sys
import subprocess as sp

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# compile
sp.check_output([sys.executable, 'make.py'])

# checkout the gh-pages branch
sp.check_output(["git", 'checkout', 'gh-pages'])

# copy built files
if os.system("cp -r ./_build/html/* ./") != 0:
    sys.exit()

# commit changes
if len(sp.check_output(["git", "diff"]).strip()) > 0:
    sp.check_output(["git", 'commit', '-a', '-m', '"automated doc upload"'])

# push
try:
    sp.check_output(["git", 'push'])
except:
    print("Could not push to gh-pages.")

# go back to master
sp.check_output(["git", 'checkout', 'master'])
