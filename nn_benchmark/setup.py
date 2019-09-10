# Code can be compiled using "python3 setup.py build_ext --inplace"

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("nn",
                             sources=["nn.pyx"],
                             language_level=3,
                             language="c++",
                             extra_compile_args=["-std=c++14", "-O3", "-march=native", "-fno-wrapv", "-DNDEBUG"],
                             include_dirs=[numpy.get_include(), "/usr/local/Cellar/openblas/0.3.7/include"])]
)
