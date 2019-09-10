import cython
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "nn_cpp.cpp" namespace "nn":
    cdef void inference_cpp[T](
            T* at, const T* a, const T visc,
            const T dxidxi, const T dyidyi, const T dzidzi,
            const int itot, const int jtot, const int ktot)

ctypedef fused float_t:
    cython.float
    cython.double

@cython.boundscheck(False)
@cython.wraparound(False)
def inference(
        np.ndarray[float_t, ndim=3, mode="c"] at not None,
        np.ndarray[float_t, ndim=3, mode="c"] a not None,
        float_t visc, float_t dxidxi, float_t dyidyi, float_t dzidzi):
    cdef int ktot, jtot, itot
    ktot, jtot, itot = at.shape[0], at.shape[1], at.shape[2]
    inference_cpp(&at[0,0,0], &a[0,0,0], visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot)
    return None

