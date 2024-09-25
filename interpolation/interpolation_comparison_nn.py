import matplotlib.pyplot as plt
import numpy as np
from numba import njit


N = 3


def int_nn(a_new, a, istart, iend, is_u):
    Ns = N//2 if is_u else 0

    for ic in range(istart, iend):
        i = (ic-istart)*N + istart

        for ii in range(0, N-Ns):
            a_new[i+ii] = a[ic]

        if is_u and N%2 == 0:
            ii = N-Ns
            a_new[i+ii] = 0.5*(a[ic] + a[ic+1])
        else:
            ii = N-Ns
            a_new[i+ii] = a[ic+1]

        for ii in range(N-Ns+1, N):
            a_new[i+ii] = a[ic+1]


dx = 200
xsize = 3200


## U
x = np.arange(0, xsize+dx/2, dx)
u = np.sin(2*(2*np.pi)/xsize * x) + 0.5*np.sin(5*(2*np.pi)/xsize * x)

dx_new = dx / N

x_int = np.arange(0, xsize+dx_new/2, dx_new)
u_int = np.zeros_like(x_int)

int_nn(u_int, u, 0, len(x)-1, True)

plt.figure()
plt.plot(x_int, u_int, 'C0^-')
plt.plot(x, u, 'k:')


## C
x = np.arange(-dx/2, xsize+dx, dx)
s = np.sin(2*(2*np.pi)/xsize * x) + 0.5*np.sin(5*(2*np.pi)/xsize * x)

dx_new = dx / N

x_int = np.arange(-dx_new/2, xsize+dx_new, dx_new)
s_int = np.zeros_like(x_int)

int_nn(s_int, s, 1, len(x)-1, False)

plt.figure()
plt.plot(x_int[1:-1], s_int[1:-1], 'C0^-')
plt.plot(x, s, 'k:')


plt.show()
