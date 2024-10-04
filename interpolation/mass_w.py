import matplotlib.pyplot as plt
import numpy as np
from numba import njit


N = 3


def int_lin(a_new, a, istart, iend):
    for ic in range(istart, iend):
        if N%2 == 0:
            i = (ic-istart)*N + istart + N//2

            for ii in range(1, N//2+1):
                fac = 1 - (ii/N) + 1/(2*N)
                a_new[i-ii  ] = fac*a[ic] + (1-fac)*a[ic-1]
                a_new[i+ii-1] = fac*a[ic] + (1-fac)*a[ic+1]

        else:
            i = (ic-istart)*N + istart + N//2

            a_new[i] = a[ic]
            for ii in range(1, N//2+1):
                fac = 1 - ii/N
                a_new[i-ii] = fac*a[ic] + (1-fac)*a[ic-1]
                a_new[i+ii] = fac*a[ic] + (1-fac)*a[ic+1]

def int_nn(a_new, a, istart, iend):
    for ic in range(istart, iend):
        if N%2 == 0:
            i = (ic-istart)*N + istart + N//2

            for ii in range(1, N//2+1):
                fac = 1# - (ii/N) + 1/(2*N)
                a_new[i-ii  ] = fac*a[ic] + (1-fac)*a[ic-1]
                a_new[i+ii-1] = fac*a[ic] + (1-fac)*a[ic+1]

        else:
            i = (ic-istart)*N + istart + N//2

            a_new[i] = a[ic]
            for ii in range(1, N//2+1):
                fac = 1# - ii/N
                a_new[i-ii] = fac*a[ic] + (1-fac)*a[ic-1]
                a_new[i+ii] = fac*a[ic] + (1-fac)*a[ic+1]




dx = 100
xsize = 3200

## C
x = np.arange(-dx/2, xsize+dx, dx)
w = np.sin(10*(2*np.pi)/xsize * x) + 0.5*np.sin(13*(2*np.pi)/xsize * x)

dx_new = dx / N

x_int = np.arange(-dx_new/2, xsize+dx_new, dx_new)

w_nn = np.empty_like(x_int)
w_nn[:] = np.nan
int_nn(w_nn, w, 1, len(x)-1)

w_lin = np.empty_like(x_int)
w_lin[:] = np.nan
int_lin(w_lin, w, 1, len(x)-1)

w_lin_mean = w_lin.copy()
for i in range(len(w)-2):
    istart = N*i + 1
    iend = istart + N
    w_lin_mean[istart:iend] = w_lin_mean[istart:iend].mean()


plt.figure()
plt.plot(x_int[1:-1], w_nn[1:-1])
# plt.plot(x_int[1:-1], w_lin[1:-1])
plt.plot(x_int[1:-1], w_lin_mean[1:-1])
plt.plot(x, w, 'k:')
plt.title('c')


plt.show()
