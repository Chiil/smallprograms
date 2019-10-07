import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import numpy as np
import kernel

# Settings
float_type = np.float32
itot, jtot, ktot = 3, 4, 5
# End settings

a = np.random.random_sample((ktot, jtot, itot))
a = a.astype(np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

a_gpu_ptr = np.frombuffer(a_gpu.as_buffer(a.nbytes), dtype=float_type)
kernel.doublify(a_gpu_ptr, itot, jtot, ktot)

a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)

print(a)
print()
print(a_doubled)
