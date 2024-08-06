import numpy as np
from numba import jit, prange
from timeit import default_timer as timer

def diff(at, a, visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot):
    at[1:ktot-1, 1:jtot-1, 1:itot-1] += visc * (
            + ( (a[1:ktot-1, 1:jtot-1, 2:itot  ] - a[1:ktot-1, 1:jtot-1, 1:itot-1])
              - (a[1:ktot-1, 1:jtot-1, 1:itot-1] - a[1:ktot-1, 1:jtot-1, 0:itot-2]) ) * dxidxi
            + ( (a[1:ktot-1, 2:jtot  , 1:itot-1] - a[1:ktot-1, 1:jtot-1, 1:itot-1])
              - (a[1:ktot-1, 1:jtot-1, 1:itot-1] - a[1:ktot-1, 0:jtot-2, 1:itot-1]) ) * dyidyi
            + ( (a[2:ktot  , 1:jtot-1, 1:itot-1] - a[1:ktot-1, 1:jtot-1, 1:itot-1])
              - (a[1:ktot-1, 1:jtot-1, 1:itot-1] - a[0:ktot-2, 1:jtot-1, 1:itot-1]) ) * dzidzi
            )

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
diff(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot)
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//2]))

# Time the loop
start = timer()
for i in range(nloop):
    diff(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot)
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))
