# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
	ext_modules = cythonize("color_eq.pyx", annotate=True),
	include_dirs = [numpy.get_include()],
)
