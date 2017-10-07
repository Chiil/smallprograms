from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("diff_f",
                             sources=["diff_f.pyx"],
                             language="c++",
                             extra_compile_args=["-Ofast", "-march=native", "-mtune=native", "-fno-wrapv"],
                             include_dirs=[numpy.get_include()])],
)
