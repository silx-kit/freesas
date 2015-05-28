import sys, glob, os
from os.path import dirname, abspath, join
base = dirname(dirname(abspath(__file__)))
libdirs = glob.glob(join(base,"build","lib*"))
if not libdirs: 
    #let's compile cython extensions
    os.system("cd %s;%s setup.py build; cd -"%(base, sys.executable)) 
    libdirs = glob.glob(join(base,"build","lib*"))
libdir = libdirs[0] 
print(libdir)
if base not in sys.modules:
    sys.path.insert(0, libdir)
