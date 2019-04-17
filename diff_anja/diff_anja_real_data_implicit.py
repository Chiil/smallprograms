#from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.optimize import minimize, approx_fprime

# DIFF
nx, ny, nz = 2304, 576, 144
xsize = 18.84955592153876
ysize = 4.71238898038469
zsize = 1.

nc_file = nc.Dataset('moser600.default.0036000.nc', 'r')
u  = nc_file.variables['u'][:,:].mean(axis=0)
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
c1 = np.zeros(c0.shape)

kappa = 0.4
ustar = 0.005
Ky0 = kappa*z*ustar
Kz0 = kappa*zh*ustar

# Distance to cover.
dx_tot = 1.578977465629577637

def mass(c):
    return (c*dy*dz[:,None]).sum()

def tdma(sol, a, b, c, nk):
    work2d = b[0,:]
    sol[0,:] /= work2d[:]

    work3d = np.empty(sol.shape)

    for k in range(1, nk):
        work3d[k,:] = c[k-1] / work2d[:]
        work2d[:] = b[k,:] - a[k,:]*work3d[k,:]
        sol[k,:] -= a[k,:]*sol[k-1,:]
        sol[k,:] /= work2d[:]

    for k in range(nk-2, -1, -1):
        sol[k,:] -= work3d[k+1,:]*sol[k+1,:]

def solve_diff(K_vect):
    Ky = K_vect[:nz]
    Kz = K_vect[nz:]

    # IMPLICIT SOLVER
    c = c0.copy()

    n_tot = 1
    dx_step = dx_tot / n_tot

    print("Solving in {:} steps.".format(n_tot))
    x_step = 0.
    c = np.fft.rfft(c, axis=1)

    print("iter: ", 0, "mass: ", (dy*dz*c[:,0]).sum(), "mass_ref: ", mass(c0))

    for n in range(n_tot):
        am  = np.zeros((nz+2, ny//2+1))
        ac  = np.zeros((nz+2, ny//2+1))
        ap  = np.zeros(nz+2)
        rhs = np.zeros((nz+2, ny//2+1), dtype=np.complex)

        # Solve all wave numbers at once.
        jj = np.arange(ny//2+1) / ny
        am[1:-1,:] = dx_step/u[:,None]*Kz[:-1,None]/(dz[:,None]*dzh[:-1,None])
        ac[1:-1,:] = dx_step/u[:,None]*Ky[:,None]/dy**2 * (2.*np.cos(2.*np.pi*jj[None,:]) - 2.) \
                   - dx_step/u[:,None]*Kz[1:,None]/(dz[:,None]*dzh[1:,None]) \
                   - dx_step/u[:,None]*Kz[:-1,None]/(dz[:,None]*dzh[:-1,None]) - 1.
        ap[1:-1] = dx_step/u*Kz[1:]/(dz*dzh[1:])
        rhs[1:-1,:] = -c[:,:]

        # Set the BC (no gradient, dirichlet = 0 at top for wavenumber 0)
        ac[0,:] = -1.
        ap[0] = 1.

        ac[-1, :] =  1. # Dirichlet and Neumann
        am[-1,1:] = -1. # Neumann
        #am[-1,:] = -1. # Neumann, ignore dirichlet

        tdma(rhs, am, ac, ap, nz+2)

        c[:,:] = rhs[1:-1,:]

        print("iter: ", n+1, "mass: ", (dy*dz*c[:,0]).sum(), "mass_ref: ", mass(np.fft.irfft(c, axis=1)))

        x_step += dx_step

    c1[:,:] = np.fft.irfft(c, axis=1)
    error = (dx*dz[:,None]*c1-dx*dz[:,None]*slice_1).sum()
    return error

K = np.append(Ky0, Kz0)
error = solve_diff(K)
print(error)

#jac = approx_fprime(K, solve_diff, 1e-3)
#print(jac)
#minimize(solve_diff, K, jac=jac, method="Newton-CG")

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
