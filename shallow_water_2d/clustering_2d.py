import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pyfftw.interfaces.numpy_fft as fft

nx = 256
ny = 256

Lx = 1e6
Ly = 1e6

# Settings
nu = 500.
c0 = 2.e-6
T_relax = 10.*86400.
c1 = 1./T_relax

x = np.arange(0., Lx, Lx/nx)
y = np.arange(0., Ly, Ly/ny)

x_km = x/1000
y_km = y/1000

kx = 2.*np.pi/Lx * np.arange(0, nx//2+1)
ky = np.zeros(ny)
ky[0:ny//2+1] = 2.*np.pi/Ly * np.arange(0, ny//2+1)
for j in range(1, ny//2+1):
    ky[-j] = -ky[j]

## RANDOM FIELD
def generate_random_field(a_std):
    # the mean size of the perturbation.
    k1 = nx//8
    
    # the variance in wave number of perturbation
    ks2 = 2.**2.
    
    arnd = 2.*np.pi*np.random.rand(ny, nx//2+1)
    afftrnd = np.cos(arnd) + 1j*np.sin(arnd)
    # calculate the radial wave numbers
    l = np.zeros(afftrnd.shape)
    for j in range(0, ny//2+1):
        for i in range(0, nx//2+1):
            l[j,i] = (i**2 + j**2)**.5
    for j in range(ny//2+1,ny):
        for i in range(0, ny//2+1):
            l[j,i] = (i**2 + (ny-j)**2)**.5
    
    # filter on radial wave number using a gaussian function
    #fftfilter = zeros(sfft.shape, dtype=np.complex)
    factor = np.exp(-(l-k1)**2 / (2.*ks2))
    
    # create the filtered field
    afft = factor*afftrnd
    
    # make sure the mean is exactly 0
    afft[0,0] = 0.
    
    a = fft.irfft2(afft)
    
    # normalize the variance to 1
    a *= a_std/np.std(a)
    return a

#h = generate_random_field(1.) + h0
q = generate_random_field(1.)

nt = 24*1000
dt = 3600.
t = 0.
n_out = 24

# Set all variables in Fourier space.
q = fft.rfft2(q)

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

def calc_rhs(q):
    q_tend = fft.rfft2(c0*np.tanh(fft.irfft2(q))) \
           - c1*q \
           - nu * (kx[np.newaxis,:]**2 + ky[:,np.newaxis]**2) * q

    return q_tend

output = True
for n in range(nt):
    if (output and n%n_out == 0):
        q_plot = fft.irfft2(q)

        print('{0}, var = '.format(n//n_out), q_plot.var())

        plt.figure()
        plt.pcolormesh(x_km, y_km, q_plot, vmin=-2, vmax=2, cmap=plt.cm.RdBu)
        plt.colorbar()
        plt.title('{0} d'.format(n*dt/86400))
        plt.tight_layout()
        plt.savefig('figs/{0:08d}.png'.format(n//n_out), dpi=100)
        plt.close()

    q_tend1 = calc_rhs(q)
    q_tend2 = calc_rhs(q + dt*q_tend1/2)
    q_tend3 = calc_rhs(q + dt*q_tend2/2)
    q_tend4 = calc_rhs(q + dt*q_tend3  )

    q += dt * (q_tend1 + 2.*q_tend2 + 2.*q_tend3 + q_tend4) / 6.

