import numpy as np
from timeit import default_timer as timer
import nn

float_type = np.float64
n0 = 375
n1 = 80

nloop = 100;
itot = 96;
jtot = 48;
ktot = 64;

ut = np.zeros((ktot, jtot, itot), dtype=float_type)
vt = np.zeros((ktot, jtot, itot), dtype=float_type)
wt = np.zeros((ktot, jtot, itot), dtype=float_type)

u = np.empty((ktot, jtot, itot), dtype=float_type)
v = np.empty((ktot, jtot, itot), dtype=float_type)
w = np.empty((ktot, jtot, itot), dtype=float_type)
u[:,:,:] = np.random.random_sample((ktot, jtot, itot))
v[:,:,:] = np.random.random_sample((ktot, jtot, itot))
w[:,:,:] = np.random.random_sample((ktot, jtot, itot))

M0 = np.random.random_sample((n1, n0))
b0 = np.random.random_sample((n1))

# Time the loop
start = timer()
for i in range(nloop):
    nn.inference(ut, vt, wt, u, v, w, M0, b0)
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))

