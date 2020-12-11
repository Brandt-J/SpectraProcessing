# -*- coding: utf-8 -*-
"""
GEPARD - Gepard-Enabled PARticle Detection
Copyright (C) 2018  Lars Bittrich and Josef Brandt, Leibniz-Institut für 
Polymerforschung Dresden e. V. <bittrich-lars@ipfdd.de>    

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program, see COPYING.  
If not, see <https://www.gnu.org/licenses/>.
"""

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import sys

if len(sys.argv) == 1:
    sys.argv.append("build_ext")
    sys.argv.append("--inplace")

ext = Extension("corrCoeff", ["corrCoeff.pyx"],
        extra_compile_args=['-O3'],)

setup(
    name="optimized calculation of correlation coefficient",
    ext_modules=cythonize([ext], annotate=True),  # accepts a glob pattern
    include_dirs=[np.get_include()]
)
