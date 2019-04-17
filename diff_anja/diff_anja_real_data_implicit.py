from numba import jit
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

u = 0.11*np.ones(z.shape)
kappa = 0.4
ustar = 0.005
Ky0 = kappa*z*ustar
Kz0 = kappa*zh*ustar

# Distance to cover.
dx_tot = 1.578977465629577637

#@jit(nopython=True)
def tdma(xc, ac, bc, cc, dc, nf):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    for k in range(1, nf):
        mc = ac[k-1]/bc[k-1]
        bc[k] = bc[k] - mc*cc[k-1]
        dc[k] = dc[k] - mc*dc[k-1]

    for k in range(nf):
        xc[k] = bc[k]

    xc[nf-1] = dc[nf-1]/bc[nf-1]

    for k in range(nf-2, -1, -1):
        xc[k] = (dc[k]-cc[k]*xc[k+1])/bc[k]

    return xc

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

    for n in range(n_tot):
        c_sol = np.zeros(nz+2, dtype=np.complex)
        am    = np.zeros(nz+2)
        ac    = np.zeros(nz+2)
        ap    = np.zeros(nz+2)
        rhs   = np.zeros(nz+2, dtype=np.complex)

        # Solve per wavenumber
        for j in range(ny//2+1):
            am[1:-1] = dx_step/u*Kz[:-1]/(dz*dzh[:-1])
            ac[1:-1] = dx_step/u*Ky/dy**2 * (2*np.cos(2.*np.pi*j/ny) - 2.) - dx_step/u*Kz[1:]/(dz*dzh[1:]) - dx_step/u*Kz[:-1]/(dz*dzh[:-1])
            ap[1:-1] = dx_step/u*Kz[1:]/(dz*dzh[1:])
            rhs[1:-1] = c[:,j]

            # Set the BC (no gradient)
            ac[0] = 1.
            ap[0] = -1.
            rhs[0] = 0.

            if (j == 0):
                ac[-1] = 1.
                rhs[-1] = 0.

            tdma(c_sol, am, ac, ap, rhs, nz+2)

            c[:,j] = c_sol[1:-1]

        #c[1:-1,1:-1] = u[:,None]/dx_step * c[1:-1,1:-1] \
        #             + Ky[:,None] * (c[1:-1,:-2] - 2*c[1:-1,1:-1] + c[1:-1,2:]) / dy**2 \
        #             + ( ( Kz[1:,None] * (c[2:,1:-1] - c[1:-1,1:-1]) / dzh[1:,None] )
        #               - ( Kz[:-1,None] * (c[1:-1,1:-1] - c[:-2,1:-1]) / dzh[:-1,None] ) ) / dz[:,None]

        x_step += dx_step

    c1[:,:] = np.fft.irfft(c)
    error = (c1-slice_1).sum()
    print(error)
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
