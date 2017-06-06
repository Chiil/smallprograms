import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pyfftw.interfaces.numpy_fft as fft

nx = 256
ny = 256

Lx = 4e6
Ly = 4e6

h0 = 100.
g = 9.81
nu = 0.
f0 = 5e-4

if (f0 > 0.):
    print('Rossby radius = {0} km'.format(1.e-3*(g*h0)**.5/f0))

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
h = np.zeros((ny, nx)) + h0

#nx_waves = 2.
#ny_waves = 2.

#h = np.sin(nx_waves*2.*np.pi*x[np.newaxis,:]/Lx) \
#  * np.sin(ny_waves*2.*np.pi*y[:,np.newaxis]/Ly)

radius = .5e6
sigma = radius/6
h = np.exp( - (x[np.newaxis,:]-Lx/2)**2 / (2.*sigma**2) - (y[:,np.newaxis]-Ly/2)**2 / (2.*sigma**2) )
h += h0

## RANDOM FIELD
def generate_random_field(a_std):
    # the mean size of the perturbation.
    k1 = nx//16
    
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
    
    # create a line for plotting with the spectral bands
    #factork = linspace(0., 25., 1000)
    #factory1 = fac1*exp(-(factork-k1)**2. / (2.*ks2))
    #factory2 = fac2*exp(-(factork-k2)**2. / (2.*ks2))
    
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

nt = 172800
dt = 100.
t = 0.

# Set all variables in Fourier space.
u = fft.rfft2(u)
v = fft.rfft2(v)
h = fft.rfft2(h)
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

#def calc_prod(a, b):
#    return fft.rfft2( fft.irfft2(a) * fft.irfft2(b) )

def calc_rhs(u, v, h, q):
    u_tend = - 1j * kx[np.newaxis,:] * calc_prod(u, u) \
             - 1j * ky[:,np.newaxis] * calc_prod(v, u) \
             - g * 1j * kx[np.newaxis,:] * h \
             + f0 * v \
             #- nu * (kx[np.newaxis,:]**2 + ky[:,np.newaxis]**2) * u
    v_tend = - 1j * kx[np.newaxis,:] * calc_prod(u, v) \
             - 1j * ky[:,np.newaxis] * calc_prod(v, v) \
             - g * 1j * ky[:,np.newaxis] * h \
             - f0 * u \
             #- nu * (kx[np.newaxis,:]**2 + ky[:,np.newaxis]**2) * v
    h_tend = - 1j * kx[np.newaxis,:] * calc_prod(u, h) \
             - 1j * ky[:,np.newaxis] * calc_prod(v, h) \
             - calc_prod(h, 1j * kx[np.newaxis,:] * u + 1j * ky[:,np.newaxis] * v) \
             #- nu * (kx[np.newaxis,:]**2 + ky[:,np.newaxis]**2) * h
             #+ fft.rfft2((200./86400)*fft.irfft2(q)) \

    q_tend = - 1j * kx[np.newaxis,:] * calc_prod(u, q) \
             - 1j * ky[:,np.newaxis] * calc_prod(v, q) \
             #- nu * (kx[np.newaxis,:]**2 + ky[:,np.newaxis]**2) * q

    return u_tend, v_tend, h_tend, q_tend

output = True
n_out = 1
for n in range(nt):
    if (output and n%n_out == 0):
        h_plot = fft.irfft2(h)-h0
        q_plot = fft.irfft2(q)

        print('{0}'.format(n//n_out))

        plot_grid = gs.GridSpec(3,1)
        plt.figure(figsize = (6,9))
        plt.subplot(plot_grid[0:2])
        plt.pcolormesh(x_km, y_km, h_plot, vmin=-0.3, vmax=0.6)
        xx, yy = np.meshgrid(x/1000., y/1000)
        nq=4
        plt.quiver(xx[::nq, ::nq], yy[::nq, ::nq], fft.irfft2(u)[::nq, ::nq], fft.irfft2(v)[::nq, ::nq], scale=2., pivot='mid')
        #plt.pcolormesh(x_km, y_km, q_plot, vmin=-2, vmax=2)
        #plt.pcolormesh(x_km, y_km, q_plot)
        #plt.colorbar()
        plt.title('{0} d'.format(n*dt/86400))
        plt.subplot(plot_grid[2])
        plt.plot(x_km, h_plot[ny//2,:], label='{0}'.format(n//n_out))
        plt.ylim(-0.3, 1.1)
        plt.grid()
        plt.tight_layout()
        plt.savefig('figs/{0:08d}.png'.format(n//n_out), dpi=100)
        plt.close()

    u_tend1, v_tend1, h_tend1, q_tend1 = calc_rhs(u, v, h, q)
    u_tend2, v_tend2, h_tend2, q_tend2 = calc_rhs(u + dt*u_tend1/2, v + dt*v_tend1/2, h + dt*h_tend1/2, q + dt*q_tend1/2)
    u_tend3, v_tend3, h_tend3, q_tend3 = calc_rhs(u + dt*u_tend2/2, v + dt*v_tend2/2, h + dt*h_tend2/2, q + dt*q_tend2/2)
    u_tend4, v_tend4, h_tend4, q_tend4 = calc_rhs(u + dt*u_tend3  , v + dt*v_tend3  , h + dt*h_tend3  , q + dt*q_tend3  )

    u += dt * (u_tend1 + 2.*u_tend2 + 2.*u_tend3 + u_tend4) / 6.
    v += dt * (v_tend1 + 2.*v_tend2 + 2.*v_tend3 + v_tend4) / 6.
    h += dt * (h_tend1 + 2.*h_tend2 + 2.*h_tend3 + h_tend4) / 6.
    q += dt * (q_tend1 + 2.*q_tend2 + 2.*q_tend3 + q_tend4) / 6.

