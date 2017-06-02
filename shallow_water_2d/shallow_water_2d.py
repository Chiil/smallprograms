import numpy as np
import matplotlib.pyplot as plt
import pyfftw.interfaces.numpy_fft as fft

nx = 96
ny = 96

Lx = 1e6
Ly = 1e6

h0 = 100.
g = 9.81
nu = 0.
f0 = 0.e-4

x = np.arange(0., Lx, Lx/nx)
y = np.arange(0., Ly, Ly/ny)

x_km = x/1000
y_km = y/1000

kx = 2.*np.pi/Lx * np.arange(0, nx//2+1)
ky = np.zeros(ny)
ky[0:ny//2+1] = 2.*np.pi/Ly * np.arange(0, ny//2+1)
for j in range(1, ny//2+1):
    ky[-j] = -ky[j]

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

#nx_waves = 2.
#ny_waves = 2.

#h = np.sin(nx_waves*2.*np.pi*x[np.newaxis,:]/Lx) \
#  * np.sin(ny_waves*2.*np.pi*y[:,np.newaxis]/Ly)

#sigma = 5e4
#h = np.exp( - (x[np.newaxis,:]-Lx/2)**2 / (2.*sigma**2) - (y[:,np.newaxis]-Ly/2)**2 / (2.*sigma**2) )

## RANDOM FIELD
# the mean wavenumber of perturbation 1
k1 = 6.

# the variance in wave number of perturbation
ks2 = 2.**2.

hrnd = 2.*np.pi*np.random.rand(ny, nx//2+1)
hfftrnd = np.cos(hrnd) + 1j*np.sin(hrnd)
# calculate the radial wave numbers
l = np.zeros(hfftrnd.shape)
for j in range(0, ny//2+1):
  for i in range(0, nx//2+1):
    l[j,i] = (i**2 + j**2)**.5
for j in range(ny//2+1,ny):
  for i in range(0, ny//2+1):
    l[j,i] = (i**2 + (ny-j)**2)**.5

# filter on radial wave number using a gaussian function
#fftfilter = zeros(sfft.shape, dtype=np.complex)
factor = np.exp(-(l-k1)**2 / (2.*ks2))

# create a line for plotting with the spectral bands
#factork = linspace(0., 25., 1000)
#factory1 = fac1*exp(-(factork-k1)**2. / (2.*ks2))
#factory2 = fac2*exp(-(factork-k2)**2. / (2.*ks2))

# create the filtered field
hfft = factor*hfftrnd

# make sure the mean is exactly 0
hfft[0,0] = 0.

h = fft.irfft2(hfft)

# normalize the variance to 1
h /= np.std(h)
# END RANDOM

h += h0

nt = 500
dt = 100.
t = 0.

plt.close('all')

plt.figure()
plt.pcolormesh(x_km, y_km, h-h0, vmin=-2, vmax=2)
plt.colorbar()
plt.title('{0}'.format(t))
plt.tight_layout()

# Set all variables in Fourier space.
u = fft.rfft2(u)
v = fft.rfft2(v)
h = fft.rfft2(h)

def pad(a):
    a_pad = np.zeros((3*ny//2, 3*nx//4+1), dtype=np.complex)
    a_pad[:ny//2,:nx//2+1] = a[:ny//2,:]
    a_pad[ny:3*ny//2,:nx//2+1] = a[ny//2:,:]
    return (9/4)*a_pad

def unpad(a_pad):
    a = np.zeros((ny, nx//2+1), dtype=complex)
    a[:ny//2,:] = a_pad[:ny//2,:nx//2+1]
    a[ny//2:,:] = a_pad[ny:3*ny//2,:nx//2+1]
    return (4/9)*a

def calc_prod(a, b):
    a_pad = pad(a)
    b_pad = pad(b)

    ab_pad = fft.rfft2( fft.irfft2(a_pad) * fft.irfft2(b_pad) )
    return unpad(ab_pad)

#def calc_prod(a, b):
#    return fft.rfft2( fft.irfft2(a) * fft.irfft2(b) )

def calc_rhs(u, v, h):
    dudx = 1j * kx[np.newaxis,:] * u
    dudy = 1j * ky[:,np.newaxis] * u
    dvdx = 1j * kx[np.newaxis,:] * v
    dvdy = 1j * ky[:,np.newaxis] * v
    dhdx = 1j * kx[np.newaxis,:] * h
    dhdy = 1j * ky[:,np.newaxis] * h

    d2udx2 = -kx[np.newaxis,:]**2 * u
    d2udy2 = -ky[:,np.newaxis]**2 * u
    d2vdx2 = -kx[np.newaxis,:]**2 * v
    d2vdy2 = -ky[:,np.newaxis]**2 * v
    d2hdx2 = -kx[np.newaxis,:]**2 * h
    d2hdy2 = -ky[:,np.newaxis]**2 * h

    u_tend = - calc_prod(u, dudx) - calc_prod(v, dudy) - g*dhdx + nu*(d2udx2 + d2udy2) + f0 * v
    v_tend = - calc_prod(u, dvdx) - calc_prod(v, dvdy) - g*dhdy + nu*(d2vdx2 + d2vdy2) - f0 * u
    h_tend = - calc_prod(u, dhdx) - calc_prod(v, dhdy) - calc_prod(h, dudx + dvdy) + nu*(d2hdx2 + d2hdy2)

    return u_tend, v_tend, h_tend

output = False
for n in range(nt):
    u_tend1, v_tend1, h_tend1 = calc_rhs(u, v, h)
    u_tend2, v_tend2, h_tend2 = calc_rhs(u + dt*u_tend1/2, v + dt*v_tend1/2, h + dt*h_tend1/2)
    u_tend3, v_tend3, h_tend3 = calc_rhs(u + dt*u_tend2/2, v + dt*v_tend2/2, h + dt*h_tend2/2)
    u_tend4, v_tend4, h_tend4 = calc_rhs(u + dt*u_tend3  , v + dt*v_tend3  , h + dt*h_tend3  )

    u += dt * (u_tend1 + 2.*u_tend2 + 2.*u_tend3 + u_tend4) / 6.
    v += dt * (v_tend1 + 2.*v_tend2 + 2.*v_tend3 + v_tend4) / 6.
    h += dt * (h_tend1 + 2.*h_tend2 + 2.*h_tend3 + h_tend4) / 6.

    if (output):
        plt.figure()
        plt.pcolormesh(x_km, y_km, fft.irfft2(h)-h0, vmin=-2., vmax=2.)
        plt.colorbar()
        plt.title('{0}'.format(t))
        plt.savefig('figs/{0:08d}.png'.format(n), dpi=100)
        plt.close()

    t += dt

# Set the variables back to physical space.
u = fft.irfft2(u)
v = fft.irfft2(v)
h = fft.irfft2(h)

plt.figure()
plt.pcolormesh(x_km, y_km, h-h0, vmin=-2., vmax=2.)
plt.colorbar()
plt.title('{0}'.format(t))
plt.show()

