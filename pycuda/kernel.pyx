import cython
import numpy as np
cimport numpy as np

cdef extern from "kernel_gpu.cu":
    cdef void launch_doublify[T](
            T* a,
            const int itot, const int jtot, const int ktot)

ctypedef fused float_t:
    cython.float
    cython.double

@cython.boundscheck(False)
@cython.wraparound(False)
def doublify(float_t[:] a, int itot, int jtot, int ktot):
    launch_doublify(&a[0], itot, jtot, ktot)
    return None

