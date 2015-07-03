#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import glob
import numpy

cy_mod = Extension("freesas._distance",
                   sources=["freesas/_distance.pyx"],
                   language="c",
                   extra_compile_args=["-fopenmp"],
                   extra_link_args=["-fopenmp"],
                   include_dirs=[numpy.get_include()])

script_files = glob.glob("scripts/*.py")

setup(name="freesas",
      version="0.2",
      author="Guillaume Bonamis, Jerome Kieffer",
      author_email="jerome.kieffer@esrf.fr",
      description="Free tools to analyze Small angle scattering data",
      packages=["freesas"],
      test_suite="test",
      data_files=glob.glob("testdata/*"),
      scripts=script_files,
      install_requires=['numpy'],
      ext_modules=[cy_mod],
      cmdclass={'build_ext': build_ext}
      )
