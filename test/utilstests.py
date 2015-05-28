import sys, glob
from os.path import dirname, abspath, join
base = dirname(dirname(abspath(__file__)))
libdir = glob.glob(join(base,"build","lib*"))[0]
print(libdir)
if base not in sys.modules:
    sys.path.insert(0, libdir)
