import numpy as np
from timeit import default_timer as timer
import nn 

nloop = 100;
itot = 384;
jtot = 384;
ktot = 384;

ut = np.zeros((ktot, jtot, itot))
vt = np.zeros((ktot, jtot, itot))
wt = np.zeros((ktot, jtot, itot))

u = np.random.random_sample((ktot, jtot, itot))
v = np.random.random_sample((ktot, jtot, itot))
w = np.random.random_sample((ktot, jtot, itot))

# Time the loop
start = timer()
for i in range(nloop):
    nn.inference(ut, vt, wt, u, v, w)
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))

