import numpy as np
import matplotlib.pyplot as plt

def init(n):
    x = np.arange(0, 100, 100/n)
    s = np.exp(-(x-20)**2 / 4**2)
    #s = np.sin(4.*np.pi/100*x)
    return x, s

def rhs(s, k, u, D):
    ds  = np.fft.irfft(1j*k*np.fft.rfft(s))
    ds2 = np.fft.irfft(-k*k*np.fft.rfft(s))
    return (-u*ds + D*ds2)

def advection_diffusion(x, s, u, D, dt):
    n = x.size
    k = 2.*np.pi/100. * np.arange(0, n/2+1)
    rk1 = rhs(s, k, u, D)
    rk2 = rhs(s + dt/2*rk1, k, u, D)
    rk3 = rhs(s + dt/2*rk2, k, u, D)
    rk4 = rhs(s + dt*rk3, k, u, D)
    s += dt/6.*(rk1 + 2*rk2 + 2*rk3 + rk4)
    
D = 0.5
u = 1
dt = 0.0125

plt.figure()
x_400, s_400 = init(400)
plt.plot(x_400, s_400, 'C0-')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('advection_diffusion_0.pdf')


plt.figure()
x_400, s_400 = init(400)
plt.plot(x_400, s_400, 'k:')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('advection_diffusion_1.pdf')


plt.figure()
x_400, s_400 = init(400)
plt.plot(x_400, s_400, 'k:')
for i in range(4000):
    advection_diffusion(x_400, s_400, u, 0, dt)
plt.plot(x_400, s_400, 'C0-', label='advec only')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('advection_diffusion_2.pdf')


plt.figure()
x_400, s_400 = init(400)
plt.plot(x_400, s_400, 'k:')
x_400, s_400 = init(400)
for i in range(4000):
    advection_diffusion(x_400, s_400, 0, D, dt)
plt.plot(x_400, s_400, 'C1-', label='diff only')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('advection_diffusion_3.pdf')


plt.figure()
x_400, s_400 = init(400)
plt.plot(x_400, s_400, 'k:')
x_400, s_400 = init(400)
for i in range(4000):
    advection_diffusion(x_400, s_400, u, D, dt)
plt.plot(x_400, s_400, 'C2-', label='advec and diff')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('advection_diffusion_4.pdf')


plt.figure()
x_400, s_400 = init(400)
plt.plot(x_400, s_400, 'k:')
for i in range(4000):
    advection_diffusion(x_400, s_400, u, 0, dt)
plt.plot(x_400, s_400, 'C0-', label='advec only')
x_400, s_400 = init(400)
for i in range(4000):
    advection_diffusion(x_400, s_400, 0, D, dt)
plt.plot(x_400, s_400, 'C1-', label='diff only')
x_400, s_400 = init(400)
for i in range(4000):
    advection_diffusion(x_400, s_400, u, D, dt)
plt.plot(x_400, s_400, 'C2-', label='advec and diff')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('advection_diffusion_5.pdf')

