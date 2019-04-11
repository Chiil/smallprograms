import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# DIFF
nx, ny, nz = 2304, 576, 144
xsize = 18.84955592153876
ysize = 4.71238898038469
zsize = 1.

nc_file = nc.Dataset('moser600.default.0036000.nc', 'r')
z  = nc_file.variables['z' ][:]
zh = nc_file.variables['zh'][:]
nc_file.close()

dx, dy = xsize/nx, ysize/ny
x = np.arange(dx/2, xsize, dx)
y = np.arange(dy/2, ysize, dy)

dz = zh[1:] - zh[:-1]
dzh = z[1:] - z[:-1]
dzh = np.append(2*z[0], dzh)
dzh = np.append(dzh, 2*(zsize-z[-1]))

slice_0 = np.fromfile('avg_ch4.01')
slice_0.shape = (nz, ny)

slice_1 = np.fromfile('avg_ch4.02')
slice_1.shape = (nz, ny)

c0 = slice_0.copy()
c0 = slice_0.copy()

u = 0.11*np.ones(z.shape)
kappa = 0.4
ustar = 0.005
Ky = kappa*z*ustar
Kz = kappa*zh*ustar

# EXPLICIT SOLVER
c = np.empty((nz+2, ny+2))
c[1:-1,1:-1] = c0[:,:]

# Distance to cover.
dx_tot = 1.578977465629577637

# Check for maximum permissible K.
dx_max_y = 0.5 * dy**2 * u / Ky
dx_max_z = 0.5 * dz**2 * u / Kz[1:]
dx_max = min( dx_max_y.min(), dx_max_z.min() )

# Keep a factor 2 safety margin.
n_tot = int( 1.5*dx_tot / dx_max )
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
                 + ( ( Kz[1:,None] * (c[2:,1:-1] - c[1:-1,1:-1]) / dzh[1:,None] )
                   - ( Kz[:-1,None] * (c[1:-1,1:-1] - c[:-2,1:-1]) / dzh[:-1,None] ) ) / dz[:,None]
    
    c[1:-1,1:-1] /= (u[:,None])/dx_step
    x_step += dx_step

print(x_step)
c1 = c[1:-1,1:-1]

plt.figure()
plt.subplot(131)
plt.pcolormesh(y, z, c0)
plt.colorbar()
plt.title('start')
plt.subplot(132)
plt.pcolormesh(y, z, c1)
plt.colorbar()
plt.title('end')
plt.subplot(133)
plt.pcolormesh(y, z, slice_1)
plt.colorbar()
plt.title('ref')
plt.tight_layout()
plt.show()

