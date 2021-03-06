# Invisic Burger equation to test effects of dealiasing.
import numpy as np
import matplotlib.pyplot as plt

# Settings.
nx = 256
nt = 140

# Solution.
x = np.arange(0., 2.*np.pi, 2.*np.pi/nx)
u = np.sin(x)

x_pad = np.arange(0., 2.*np.pi, 2.*np.pi/(3/2*nx))

k = np.arange(0, nx//2+1)
dt = 0.01

u = np.fft.rfft(u)

def pad(a):
    a_pad = np.zeros(3*nx//4+1, dtype=np.complex)
    a_pad[:nx//2+1] = a[:]
    return (3/2)*a_pad

def unpad(a_pad):
    a = np.zeros(nx//2+1, dtype=np.complex)
    a[:] = a_pad[:nx//2+1]
    return (2/3)*a

def calc_prod_nopad(a, b):
    a = np.fft.irfft(a)
    b = np.fft.irfft(b)
    return np.fft.rfft(a*b)

def calc_prod_pad(a, b):
    a = np.fft.irfft(pad(a))
    b = np.fft.irfft(pad(b))
    return unpad(np.fft.rfft(a*b))

def calc_rhs(a):
    a_grad = 1j*k*a
    a_tend = -1.*calc_prod_nopad(a, a_grad)
    return a_tend

plt.figure()
for i in range(nt):
    # RK4
    u_tend1 = calc_rhs(u)
    u_tend2 = calc_rhs(u + dt*u_tend1/2)
    u_tend3 = calc_rhs(u + dt*u_tend2/2)
    u_tend4 = calc_rhs(u + dt*u_tend3)
    u += dt * (u_tend1 + 2.*u_tend2 + 2.*u_tend3 + u_tend4) / 6.

    if (i%10 == 0):
        plt.subplot(211)
        p = plt.plot(x, np.fft.irfft(u), label='{0}'.format(i))
        plt.plot(x_pad, np.fft.irfft(pad(u)), ':', color=p[0].get_color(), label='{0}'.format(i))

        u_energy = abs(u)**2
        u_energy[1:-1] *= 2
        plt.subplot(212)
        plt.semilogx(k, k*u_energy, label='{0}'.format(i))

plt.legend(loc=0, ncol=3, frameon=False)
plt.show()

