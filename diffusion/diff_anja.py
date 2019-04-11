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

u = 0.1 * z
kappa = 0.4*10
ustar = 0.005
Ky = kappa*z*ustar
Kz = kappa*0.5*(z[:-1]+z[1:])*ustar
Kz = np.append(0., Kz)
Kz = np.append(Kz, kappa*zsize*ustar)

# EXPLICIT SOLVER
c = np.empty((nz+2, ny+2))
c[1:-1,1:-1] = c0[:,:]


# Distance to cover.
dx_tot = 1.

# Check for maximum permissible K.

dx_max_y = 0.5 * dy**2 * u / Ky
dx_max_z = 0.5 * dz**2 * u / Kz[1:]
dx_max = min( dx_max_y.min(), dx_max_z.min() )

# Keep a factor 2 safety margin.
n_tot = 2*int( dx_tot / dx_max )
dx_step = dx_tot / n_tot

print("Solving in {:} steps.".format(n_tot))
x_step = 0.
for n in range(n_tot):
    # Ghost cells in y
    c[:, 0] = c[:,-2]
    c[:,-1] = c[:, 1]
    
    # Ghost cells in z
    c[ 0,:] = c[ 1,:]
    c[-1,:] = c[-2,:]
    
    c[1:-1,1:-1] = u[:,None]/dx_step * c[1:-1,1:-1] \
                 + Ky[:,None] * (c[1:-1,:-2] - 2*c[1:-1,1:-1] + c[1:-1,2:]) / dy**2 \
                 + ( ( Kz[1:,None] * (c[2:,1:-1] - c[1:-1,1:-1]) / dz ) - ( Kz[:-1,None] * (c[1:-1,1:-1] - c[:-2,1:-1]) / dz ) ) /dz
    
    c[1:-1,1:-1] /= (u[:,None])/dx_step
    x_step += dx_step

print(x_step)
c1 = c[1:-1,1:-1]

plt.figure()
plt.subplot(121)
plt.pcolormesh(y, z, c0)
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(y, z, c1)
plt.colorbar()
plt.show()
