import cython
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "diff_cpp.cpp" namespace "diff":
    cdef void diff_cpp[T](T* at, const T* a, const T visc,
                          const T dxidxi, const T dyidyi, const T dzidzi,
                          const int itot, const int jtot, const int ktot)

@cython.boundscheck(False)
@cython.wraparound(False)
def diff(np.ndarray[float, ndim=3, mode="c"] at not None,
         np.ndarray[float, ndim=3, mode="c"] a not None,
         float visc, float dxidxi, float dyidyi, float dzidzi):
    cdef int ktot, jtot, itot
    ktot, jtot, itot = at.shape[0], at.shape[1], at.shape[2]
    diff_cpp(&at[0,0,0], &a[0,0,0], visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot)
    return None

