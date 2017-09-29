import numpy as np
import diff
from timeit import default_timer as timer

nloop = 20;
itot = 256;
jtot = 256;
ktot = 256;
ncells = itot*jtot*ktot;

at = np.zeros((ktot, jtot, itot))

index = np.arange(ncells)
a = (index/index+1)**2
a.shape = (ktot, jtot, itot)

# Check results
diff.diff(at, a, 0.1, 0.1, 0.1, 0.1)
print("at={:}".format(at[23]))

# Time the loop
start = timer()
for i in range(nloop):
    diff.diff(at, a, 0.1, 0.1, 0.1, 0.1)
end = timer()

print((end-start)/nloop)

