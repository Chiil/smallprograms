import numpy as np
import matplotlib.pyplot as plt
import pyfftw.interfaces.numpy_fft as fft

nx = 100
ny = 100

Lx = 1e6
Ly = 1e6

h0 = 100.
g = 9.81
nu = 0.
f0 = 1e-4

x = np.arange(0., Lx, Lx/nx)
y = np.arange(0., Ly, Ly/ny)

x_km = x/1000
y_km = y/1000

kx = 2.*np.pi/Lx * np.arange(0, nx//2+1)
ky = np.zeros(ny)
ky[0:ny//2+1] = 2.*np.pi/Ly * np.arange(0, ny//2+1)
for j in range(1, ny//2+1):
    ky[-j] = -ky[j]

print(ky)

nx_waves = 2.
ny_waves = 2.

#h = np.sin(nx_waves*2.*np.pi*x[np.newaxis,:]/Lx) \
#  * np.sin(ny_waves*2.*np.pi*y[:,np.newaxis]/Ly)

sigma = 5e4
h = np.exp( - (x[np.newaxis,:]-Lx/2)**2 / (2.*sigma**2) - (y[:,np.newaxis]-Ly/2)**2 / (2.*sigma**2) )
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
h += h0

nt = 2000
dt = 10.
t = 0.

plt.close('all')

#plt.figure()
#plt.pcolormesh(x_km, y_km, h)
#plt.colorbar()
#plt.title('{0}'.format(t))
#plt.tight_layout()
#plt.show()

# Set all variables in Fourier space.
u = fft.rfft2(u)
v = fft.rfft2(v)
h = fft.rfft2(h)

#def calc_prod(a, b):
#    return fft.rfft2( fft.irfft2(a) * fft.irfft2(b) )

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

for n in range(nt):
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

    u += dt * u_tend
    v += dt * v_tend
    h += dt * h_tend

    t += dt

# Set the variables back to physical space.
u = fft.irfft2(u)
v = fft.irfft2(v)
h = fft.irfft2(h)

plt.figure()
plt.subplot(211)
plt.pcolormesh(x_km, y_km, h)
plt.colorbar()
plt.title('{0}'.format(t))

plt.subplot(212)
plt.plot(x_km, h[ny//2, :], 'o')
plt.plot(y_km, h[:, nx//2], 'o')
plt.title('{0}'.format(t))
plt.tight_layout()

plt.show()

