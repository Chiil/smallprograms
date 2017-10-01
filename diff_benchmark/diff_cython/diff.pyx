# import both numpy and the Cython declarations for numpy
import cython
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "diff_cpp.cpp" namespace "diff":
    void diff_cpp(double* at, const double* a, const double visc,
                  const double dxidxi, const double dyidyi, const double dzidzi,
                  const int itot, const int jtot, const int ktot)

@cython.boundscheck(False)
@cython.wraparound(False)
def diff(np.ndarray[double, ndim=3, mode="c"] at not None,
         np.ndarray[double, ndim=3, mode="c"] a not None,
         double visc, double dxidxi, double dyidyi, double dzidzi):
    cdef int ktot, jtot, itot
    ktot, jtot, itot = at.shape[0], at.shape[1], at.shape[2]
    diff_cpp(&at[0,0,0], &a[0,0,0], visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot)
    return None

