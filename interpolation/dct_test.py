import numpy as np
from scipy.fftpack import idct, dct


def dct_rfft(a):
    aa = np.empty_like(a)
    aa_fft = np.empty_like(a)

    aa[:(N-1)//2+1] = a[::2]
    aa[(N-1)//2+1:] = a[::-2]

    aa_fft_tmp = np.fft.rfft(aa)

    k = np.arange(len(a)//2+1)

    # aa_fft_tmp *= 2 * np.exp(-1j*np.pi*k/(2*N))
    # aa_fft_tmp *= 2*np.cos(-np.pi*k/(2*N)) + 2*1j*np.sin(-np.pi*k/(2*N))
    # aa_fft[len(a)//2+1:] = - aa_fft_tmp.imag[-2:0:-1]

    aa_fft[:len(a)//2+1] = aa_fft_tmp.real * 2*np.cos(-np.pi*k/(2*N)) - aa_fft_tmp.imag * 2*np.sin(-np.pi*k/(2*N))
    aa_fft[len(a)//2+1:] = - (aa_fft_tmp.real * 2*np.sin(-np.pi*k/(2*N)) + aa_fft_tmp.imag * 2*np.cos(-np.pi*k/(2*N)))[-2:0:-1]

    return aa_fft


def idct_rfft(a_rfft):
    len_fft = len(a_rfft)//2+1

    k = np.arange(len_fft)
    # W = 0.5*np.exp(1j*2.*np.pi*k/(4*N))
    W = 0.5*np.cos(2.*np.pi*k/(4*N)) + 0.5*1j*np.sin(2.*np.pi*k/(4*N))

    a_fft = np.zeros(len_fft, dtype=np.complex64)
    a_fft[0] = W[0]*(a_rfft[0])
    for i in range(1, len_fft):
        a_fft[i] = W[i]*(a_rfft[i] - 1j*a_rfft[N-i])

    aa = np.fft.irfft(a_fft)

    a = np.empty_like(aa)

    a[::2] = aa[:(N-1)//2+1]
    a[::-2] = aa[(N-1)//2+1:]

    return a


L = 1000
N = 16
dx = L / N

x = np.arange(dx/2, L, dx)
a = np.random.rand(len(x))


a_dct_ref = dct(a)
a_dct_fft = dct_fft(a)
a_dct_rfft = dct_rfft(a)

print(abs(a_dct_fft - a_dct_ref).max())
print(abs(a_dct_rfft - a_dct_ref).max())

a_inv_ref = idct(a_dct_ref) / (2*N)
a_inv_rfft = idct_rfft(a_dct_rfft)

print(a)
print(a_inv_ref)
print(a_inv_rfft)
