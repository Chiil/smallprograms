import numpy as np
import diff
from timeit import default_timer as timer

nloop = 100;
itot = 384;
jtot = 384;
ktot = 384;
ncells = itot*jtot*ktot;

at = np.zeros((ktot, jtot, itot))

index = np.arange(ncells)
a = (index/(index+1))**2
a.shape = (ktot, jtot, itot)

# Check results
diff.diff(at.T, a.T, 0.1, 0.1, 0.1, 0.1)
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//2]))

# Time the loop
start = timer()
for i in range(nloop):
    diff.diff(at.T, a.T, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot)
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))

