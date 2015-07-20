#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import shutil
from setuptools import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import glob
import numpy

cmdclass = {'build_ext': build_ext}

cy_mod = Extension("freesas._distance",
                   sources=["freesas/_distance.pyx"],
                   language="c",
                   extra_compile_args=["-fopenmp"],
                   extra_link_args=["-fopenmp"],
                   include_dirs=[numpy.get_include()])

script_files = glob.glob("scripts/*.py")

# ################### #
# build_doc commandes #
# ################### #

try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda: False
    from sphinx.setup_command import BuildDoc
except ImportError:
    sphinx = None

if sphinx:
    class build_doc(BuildDoc):
        def run(self):

            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version

            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))

            # Build the Users Guide in HTML and TeX format
            for builder in ('html', 'latex'):
                self.builder = builder
                self.builder_target_dir = os.path.join(self.build_dir, builder)
                self.mkpath(self.builder_target_dir)
                BuildDoc.run(self)
            sys.path.pop(0)
    cmdclass['build_doc'] = build_doc


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
      cmdclass=cmdclass
      )
