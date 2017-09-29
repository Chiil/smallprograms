from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("diff",
                             sources=["diff.pyx"],
                             language="c++",
                             extra_compile_args=["-Ofast -march=native"],
                             include_dirs=[numpy.get_include()])],
)
