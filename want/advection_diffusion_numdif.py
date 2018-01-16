import numpy as np
import matplotlib.pyplot as plt

def init(n):
    x = np.arange(0, 100, 100/n)
    s = np.exp(-(x-20)**2 / 4**2)
    s = np.zeros(x.shape)
    s[20:30] = 1.
    return x, s

def rhs(s, x, u, D):
    dx = x[1] - x[0]
    ds = np.gradient(s, dx)
    ds2 = np.gradient(ds, dx)
    return (-u*ds + D*ds2)

def rhs_b(s, x, u, D):
    dx = x[1] - x[0]
    ds = np.gradient(s, dx)
    ds2 = np.gradient(ds, dx)
    ds[1:] = (s[1:]-s[:-1])/dx
    return (-u*ds + D*ds2)

def rhs_f(s, x, u, D):
    dx = x[1] - x[0]
    ds = np.gradient(s, dx)
    ds2 = np.gradient(ds, dx)
    ds[:-1] = (s[1:]-s[:-1])/dx
    return (-u*ds + D*ds2)

def advection_diffusion(x, s, u, D, dt):
    n = x.size
    rk1 = rhs(s, x, u, D)
    rk2 = rhs(s + dt/2*rk1, x, u, D)
    rk3 = rhs(s + dt/2*rk2, x, u, D)
    rk4 = rhs(s + dt*rk3, x, u, D)
    s += dt/6.*(rk1 + 2*rk2 + 2*rk3 + rk4)

def advection_diffusion_b(x, s, u, D, dt):
    n = x.size
    rk1 = rhs_b(s, x, u, D)
    rk2 = rhs_b(s + dt/2*rk1, x, u, D)
    rk3 = rhs_b(s + dt/2*rk2, x, u, D)
    rk4 = rhs_b(s + dt*rk3, x, u, D)
    s += dt/6.*(rk1 + 2*rk2 + 2*rk3 + rk4)

def advection_diffusion_f(x, s, u, D, dt):
    n = x.size
    rk1 = rhs_f(s, x, u, D)
    rk2 = rhs_f(s + dt/2*rk1, x, u, D)
    rk3 = rhs_f(s + dt/2*rk2, x, u, D)
    rk4 = rhs_f(s + dt*rk3, x, u, D)
    s += dt/6.*(rk1 + 2*rk2 + 2*rk3 + rk4)
    
D = 0.
u = 1.
dt = 0.5

plt.figure()
x_50, s_50 = init(50)
plt.plot(x_50, s_50, 'C0-')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.tight_layout()
plt.savefig('advection_diffusion_numdiff_c0.pdf')

plt.figure()
x_50, s_50 = init(50)
plt.plot(x_50, s_50, 'k:')
for i in range(10):
    advection_diffusion(x_50, s_50, u, 0, dt)
plt.plot(x_50, s_50, 'C0-')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.tight_layout()
plt.savefig('advection_diffusion_numdiff_c1.pdf')

plt.figure()
x_50, s_50 = init(50)
plt.plot(x_50, s_50, 'C1-')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.tight_layout()
plt.savefig('advection_diffusion_numdiff_b0.pdf')

plt.figure()
x_50, s_50 = init(50)
plt.plot(x_50, s_50, 'k:')
for i in range(10):
    advection_diffusion_b(x_50, s_50, u, 0, dt)
plt.plot(x_50, s_50, 'C1-')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.tight_layout()
plt.savefig('advection_diffusion_numdiff_b1.pdf')

plt.figure()
x_50, s_50 = init(50)
plt.plot(x_50, s_50, 'C2-')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.tight_layout()
plt.savefig('advection_diffusion_numdiff_f0.pdf')

plt.figure()
x_50, s_50 = init(50)
plt.plot(x_50, s_50, 'k:')
for i in range(5):
    advection_diffusion_f(x_50, s_50, u, 0, dt)
plt.plot(x_50, s_50, 'C2-')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.tight_layout()
plt.savefig('advection_diffusion_numdiff_f1.pdf')

plt.figure()
x_50, s_50 = init(50)
plt.plot(x_50, s_50, 'k:')
for i in range(5):
    advection_diffusion(x_50, s_50, u, 0, dt)
plt.plot(x_50, s_50, 'C0-')

x_50, s_50 = init(50)
for i in range(5):
    advection_diffusion_b(x_50, s_50, u, 0, dt)
plt.plot(x_50, s_50, 'C1-')

x_50, s_50 = init(50)
for i in range(5):
    advection_diffusion_f(x_50, s_50, u, 0, dt)
plt.plot(x_50, s_50, 'C2-')

plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.tight_layout()
plt.savefig('advection_diffusion_numdiff_all.pdf')

