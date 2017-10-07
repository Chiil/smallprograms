import numpy as np
from timeit import default_timer as timer

nloop = 100;
itot = 384;
jtot = 384;
ktot = 384;
ncells = itot*jtot*ktot;

#float_type = np.float32
float_type = np.float64

if (float_type == np.float32):
    import diff_f as diff
elif (float_type == np.float64):
    import diff_d as diff

at = np.zeros((ktot, jtot, itot), dtype=float_type)

index = np.arange(ncells, dtype=float_type)
a = (index/(index+1))**2
del(index)
a.shape = (ktot, jtot, itot)

# Check results
diff.diff(at, a, 0.1, 0.1, 0.1, 0.1)
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//2]))

# Time the loop
start = timer()
for i in range(nloop):
    diff.diff(at, a, 0.1, 0.1, 0.1, 0.1)
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))

