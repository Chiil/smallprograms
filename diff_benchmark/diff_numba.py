import numpy as np
from numba import jit, prange
from timeit import default_timer as timer

@jit(nopython=True, parallel=True, nogil=True)
def diff(at, a, visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot):
    for k in prange(1, ktot-1):
        for j in prange(1, jtot-1):
            for i in prange(1, itot-1):
                at[k, j, i] += visc * ( \
                        + ( (a[k+1, j  , i  ] - a[k  , j  , i  ]) \
                          - (a[k  , j  , i  ] - a[k-1, j  , i  ]) ) * dxidxi \
                        + ( (a[k  , j+1, i  ] - a[k  , j  , i  ]) \
                          - (a[k  , j  , i  ] - a[k  , j-1, i  ]) ) * dyidyi \
                        + ( (a[k  , j  , i+1] - a[k  , j  , i  ]) \
                          - (a[k  , j  , i  ] - a[k  , j  , i-1]) ) * dzidzi )

float_type = np.float32
# float_type = np.float64

nloop = 30;
itot = 384;
jtot = 384;
ktot = 384;
ncells = itot*jtot*ktot;

at = np.zeros((ktot, jtot, itot), dtype=float_type)

index = np.arange(ncells, dtype=float_type)
a = (index/(index+1))**2
del(index)
a.shape = (ktot, jtot, itot)

# Check results
c = float_type(0.1)
diff(at, a, c, c, c, c, itot, jtot, ktot)
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//2]))

# Time the loop
start = timer()
for i in range(nloop):
    diff(at, a, c, c, c, c, itot, jtot, ktot)
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))
