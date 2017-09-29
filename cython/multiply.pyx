# import both numpy and the Cython declarations for numpy
import cython
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_multiply(double* array, double value, int m, int n)

@cython.boundscheck(False)
@cython.wraparound(False)
def multiply(np.ndarray[double, ndim=2, mode="c"] input not None, double value):
    cdef int m, n
    m, n = input.shape[0], input.shape[1]
    c_multiply (&input[0,0], value, m, n)
    return None

