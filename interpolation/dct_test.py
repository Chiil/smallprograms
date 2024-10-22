import numpy as np
from scipy.fftpack import dct
from numba import njit
import matplotlib.pyplot as plt


def dct_fft(a):
    aa = np.empty_like(a)
    aa[:(N-1)//2+1] = a[::2]

    aa[(N-1)//2+1:] = a[::-2]
    aa_fft = np.fft.fft(aa)

    k = np.arange(len(a))
    aa_fft *= 2 * np.exp(-1j*np.pi*k/(2*N))

    return aa_fft.real


def dct_rfft(a):
    aa = np.empty_like(a)
    aa_fft = np.empty_like(a)

    aa[:(N-1)//2+1] = a[::2]
    aa[(N-1)//2+1:] = a[::-2]

    aa_fft_tmp = np.fft.rfft(aa)

    k = np.arange(len(a)//2+1)
    aa_fft_tmp *= 2 * np.exp(-1j*np.pi*k/(2*N))

    aa_fft[:len(a)//2+1] = aa_fft_tmp.real
    aa_fft[len(a)//2+1:] = - aa_fft_tmp.imag[-2:0:-1]

    return aa_fft


L = 1000
N = 8

dx = L / N

x = np.arange(dx/2, L, dx)
# a = 0.3*np.cos( (2*np.pi)/L*x ) + np.cos(5*(2*np.pi)/L*x )
a = np.random.rand(len(x))


a_dct_ref = dct(a)
a_dct_fft = dct_fft(a)
a_dct_rfft = dct_rfft(a)

print(abs(a_dct_fft - a_dct_ref).max())
print(abs(a_dct_rfft - a_dct_ref).max())

# plt.figure()
# plt.plot(x, a)
# plt.show()

