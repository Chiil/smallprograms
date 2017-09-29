import numpy as np
import diff

nloop = 20;
itot = 256;
jtot = 256;
ktot = 256;
ncells = itot*jtot*ktot;

a  = np.empty((ktot, jtot, itot))
at = np.empty((ktot, jtot, itot))

# Check results
diff.diff(at, a, 0.1, 0.1, 0.1, 0.1)
print("at={0.20f}".format(at[23]))

for i in range(nloop):
    diff.diff(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot)
