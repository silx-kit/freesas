#!/usr/bin/env python
from setuptools import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import glob

cy_mod = Extension("freesas._distance",
    sources=["freesas/_distance.pyx"],
    language="c",
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"])

setup(name="freesas",
      version = "0.1",
      author = "Guillaume Bonamis, Jerome Kieffer",
      author_email = "jerome.kieffer@esrf.fr",
      description = "Free tools to analyze Small angle scattering data",
      packages = ["freesas"],
      test_suite = "test",
      data_files = glob.glob("testdata/*"),
#       install_requires = ['
      ext_modules=[cy_mod],
      cmdclass={'build_ext': build_ext}
      )
