import numpy as np
import matplotlib.pyplot as plt

nx, ny, nz = 256, 192, 128

xsize = 2.*np.pi
ysize = np.pi
zsize = 1.

dx, dy, dz = xsize/nx, ysize/ny, zsize/nz

x = np.arange(dx/2, xsize, dx)
y = np.arange(dy/2, ysize, dy)
z = np.arange(dz/2, zsize, dz)

c0 = np.empty((nz, ny))

sigma_c = 0.1
c0[:,:] = np.exp(-(y[None,:]-ysize/2)**2 / sigma_c**2) * np.exp(-(z[:,None]-0.3)**2 / sigma_c**2)

plt.figure()
plt.pcolormesh(y, z, c0)
plt.colorbar()
plt.show()
