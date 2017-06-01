import numpy as np
import matplotlib.pyplot as plt
import pyfftw.interfaces.numpy_fft as fft

nx = 128
ny = 128

Lx = 1e6
Ly = 1e6

h0 = 100.
g = 9.81

x = np.arange(0., Lx, Lx/nx)
y = np.arange(0., Ly, Ly/ny)

kx = 2.*np.pi/Lx * np.arange(0, nx//2+1)
ky = np.zeros(ny)
ky[0:ny//2+1] = 2.*np.pi/Ly * np.arange(0, ny//2+1)
for j in range(1, ny//2+1):
    ky[-j] = -ky[j]

nx_waves = 2.
ny_waves = 2.

h = np.sin(nx_waves*2.*np.pi*x[np.newaxis,:]/Lx) \
        * np.sin(ny_waves*2.*np.pi*y[:,np.newaxis]/Ly)

h += h0

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

nt = 2000
dt = 10.
t = 0.

plt.close('all')
#plt.figure()
#plt.contourf(x, y, h)
#plt.colorbar()
#plt.title('{0}'.format(t))

# Set all variables in Fourier space.
u = np.fft.rfft2(u)
v = np.fft.rfft2(v)
h = np.fft.rfft2(h)

#def calc_prod(a, b):
#    return np.fft.rfft2( np.fft.irfft2(a) * np.fft.irfft2(b) )

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

    ab_pad = np.fft.rfft2( np.fft.irfft2(a_pad) * np.fft.irfft2(b_pad) )
    return unpad(ab_pad)

for n in range(nt):
    dudx = 1j * kx[np.newaxis,:] * u
    dudy = 1j * ky[:,np.newaxis] * u
    dvdx = 1j * kx[np.newaxis,:] * v
    dvdy = 1j * ky[:,np.newaxis] * v
    dhdx = 1j * kx[np.newaxis,:] * h
    dhdy = 1j * ky[:,np.newaxis] * h

    u_tend = - calc_prod(u, dudx) - calc_prod(v, dudy) - g*dhdx
    v_tend = - calc_prod(u, dvdx) - calc_prod(v, dvdy) - g*dhdy
    h_tend = - calc_prod(u, dhdx) - calc_prod(v, dhdy) - calc_prod(h, dudx + dvdy)

    u += dt * u_tend
    v += dt * v_tend
    h += dt * h_tend

    t += dt

# Set the variables back to physical space.
u = np.fft.irfft2(u)
v = np.fft.irfft2(v)
h = np.fft.irfft2(h)

x_km = x/1000
y_km = y/1000

plt.figure(figsize=(12,6))
plt.subplot(131)
plt.contourf(x_km, y_km, u)
plt.colorbar()
plt.subplot(132)
plt.contourf(x_km, y_km, v)
plt.colorbar()
plt.subplot(133)
plt.contourf(x_km, y_km, h)
plt.colorbar()
plt.title('{0}'.format(t))
plt.tight_layout()
plt.show()

