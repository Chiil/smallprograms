#!python
#cython: language_level=3

import cython
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "nn_cpp.cpp" namespace "nn":
    cdef void inference_cpp[T](
            T* ut, T* vt, T* wt,
            const T* u, const T* v, const T* w,
            const T* M0, const T* b0,
            const T* M1, const T* b1,
            const int itot, const int jtot, const int ktot)

ctypedef fused float_t:
    #cython.float
    cython.double

@cython.boundscheck(False)
@cython.wraparound(False)
def inference(
        np.ndarray[float_t, ndim=3, mode="c"] ut not None,
        np.ndarray[float_t, ndim=3, mode="c"] vt not None,
        np.ndarray[float_t, ndim=3, mode="c"] wt not None,
        np.ndarray[float_t, ndim=3, mode="c"] u not None,
        np.ndarray[float_t, ndim=3, mode="c"] v not None,
        np.ndarray[float_t, ndim=3, mode="c"] w not None,
        np.ndarray[float_t, ndim=2, mode="c"] M0 not None,
        np.ndarray[float_t, ndim=1, mode="c"] b0 not None,
        np.ndarray[float_t, ndim=2, mode="c"] M1 not None,
        np.ndarray[float_t, ndim=1, mode="c"] b1 not None):
    cdef int ktot, jtot, itot
    ktot, jtot, itot = ut.shape[0], ut.shape[1], ut.shape[2]
    inference_cpp(
            &ut[0,0,0], &vt[0,0,0], &wt[0,0,0],
            &u[0,0,0], &v[0,0,0], &w[0,0,0],
            &M0[0,0], &b0[0],
            &M1[0,0], &b1[0],
            itot, jtot, ktot)
    return None

