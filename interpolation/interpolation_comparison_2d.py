import matplotlib.pyplot as plt
import numpy as np
from numba import njit

N = 3

@njit
def int_u(a_new, a, istart, iend, jstart, jend, igc, jgc):
    for jc in range(jstart, jend):
        for ic in range(istart, iend):
            i = (ic-igc)*N + igc
            j = (jc-jgc)*N + jgc + N//2

            for jj in range(N):
                for ii in range(N):
                    fi = ii/N
                    fj = jj/N
                    a_new[j+jj, i+ii] = (1-fi)*(1-fj)*a[jc, ic] + fi*(1-fj)*a[jc, ic+1] + (1-fi)*fj*a[jc+1, ic] + fi*fj*a[jc+1, ic+1]


@njit
def int_c(a_new, a, istart, iend, jstart, jend, igc, jgc):
    for jc in range(jstart, jend):
        for ic in range(istart, iend):
            i = (ic-igc)*N + igc + N//2
            j = (jc-jgc)*N + jgc + N//2

            for jj in range(N):
                for ii in range(N):
                    fi = ii/N
                    fj = jj/N
                    a_new[j+jj, i+ii] += (1-fi)*(1-fj)*a[jc, ic] + fi*(1-fj)*a[jc, ic+1] + (1-fi)*fj*a[jc+1, ic] + fi*fj*a[jc+1, ic+1]


## Set up the grids
dx, dy = 200, 200
xsize, ysize = 3200, 3200

xh = np.arange(-dx, xsize+dx/2, dx)
x = np.arange(-dx/2, xsize+dx, dx)

yh = np.arange(-dy, ysize+dy/2, dy)
y = np.arange(-dy/2, ysize+dy, dy)

dx_new, dy_new = dx/N, dy/N

xh_new = np.arange(-dx_new, xsize+dx_new/2, dx_new)
x_new = np.arange(-dx_new/2, xsize+dx_new, dx_new)

yh_new = np.arange(-dy_new, ysize+dy_new/2, dy_new)
y_new = np.arange(-dy_new/2, ysize+dy_new, dy_new)

u_int = np.zeros((len(y_new), len(xh_new)))
s_int = np.zeros((len(y_new), len(x_new)))


## U
u = np.sin(2*(2*np.pi)/xsize * xh[None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * y[:, None])
# int_u(u_int, u, 0, len(xh)-1, 1, len(y)-1)

xh_plot = np.array([ 0, *x[1:-1], xsize])
y_plot = yh[1:]

xh_new_plot = np.array([ 0, *x_new[1:-1], xsize])
y_new_plot = yh_new[1:]

plt.figure()
plt.subplot(121)
plt.pcolormesh(xh_plot, y_plot, u[1:-1, 1:])
plt.subplot(122)
plt.pcolormesh(xh_new_plot, y_new_plot, u_int[1:-1, 1:])
plt.xlim(0, xsize)
plt.ylim(0, ysize)


## C
s = np.sin(2*(2*np.pi)/xsize * x[None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * y[:, None])
int_c(s_int, s, 1, len(x)-1, 1, len(y)-1, 1, 1)

x_plot = xh[1:]
x_new_plot = xh_new[1:]

plt.figure()
plt.subplot(121)
plt.pcolormesh(x_plot, y_plot, s[1:-1, 1:-1])
plt.subplot(122)
plt.pcolormesh(x_new_plot, y_new_plot, s_int[1:-1, 1:-1])

print(y[1], y_new[1+N//2])
plt.figure()
plt.plot(x_new, s_int[1+N//2, :])
plt.plot(x, s[1, :], 'k:')

plt.show()
