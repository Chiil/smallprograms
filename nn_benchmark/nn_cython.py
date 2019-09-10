import numpy as np
from timeit import default_timer as timer
import nn 

#float_type = np.float32
float_type = np.float64

nloop = 100;
itot = 384;
jtot = 384;
ktot = 384;

at = np.zeros((ktot, jtot, itot), dtype=float_type)
a = np.random.rand(ktot, jtot, itot, dtype=float_type)

# Time the loop
start = timer()
for i in range(nloop):
    nn.inference(at, a, 0.1, 0.1, 0.1, 0.1)
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))

