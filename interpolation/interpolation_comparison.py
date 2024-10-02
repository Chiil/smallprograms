import matplotlib.pyplot as plt
import numpy as np
from numba import njit


N = 6


# @njit
def int_c(a_new, a, istart, iend):
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


# @njit
def int_u(a_new, a, istart, iend):
    for ic in range(istart, iend):
        i = (ic-istart)*N + istart

        for ii in range(0, N):
            fac = 1 - ii/N
            a_new[i+ii] = fac*a[ic] + (1-fac)*a[ic+1]


dx = 200
xsize = 3200


## U
x = np.arange(0, xsize+dx/2, dx)
u = np.sin(2*(2*np.pi)/xsize * x) + 0.5*np.sin(5*(2*np.pi)/xsize * x)

dx_new = dx / N

x_int = np.arange(0, xsize+dx_new/2, dx_new)
u_int = np.zeros_like(x_int)

int_u(u_int, u, 0, len(x)-1)

plt.figure()
plt.plot(x_int, u_int)
plt.plot(x, u, 'k:')
plt.title('u')


## C
x = np.arange(-dx/2, xsize+dx, dx)
s = np.sin(2*(2*np.pi)/xsize * x) + 0.5*np.sin(5*(2*np.pi)/xsize * x)

dx_new = dx / N

x_int = np.arange(-dx_new/2, xsize+dx_new, dx_new)
s_int = np.zeros_like(x_int)

int_c(s_int, s, 1, len(x)-1)

plt.figure()
plt.plot(x_int[1:-1], s_int[1:-1])
plt.plot(x, s, 'k:')
plt.title('c')


plt.show()
