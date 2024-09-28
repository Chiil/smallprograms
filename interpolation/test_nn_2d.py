import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from enum import Enum

Ni = 4
Nj = 4

Loc = Enum('Loc', ['C', 'U', 'V', 'W'])


@njit
def interpolate(a_new, a, istart, iend, jstart, jend, igc, jgc, loc):
    Nsi = Ni // 2 if loc == Loc.U else 0
    Nsj = Nj // 2 if loc == Loc.V else 0

    for jc in range(jstart, jend):
        for ic in range(istart, iend):
            i = (ic-igc)*Ni + igc
            j = (jc-jgc)*Nj + jgc

            if loc == Loc.C or loc == Loc.W:
                for jj in range(0, Nj):
                    for ii in range(0, Ni):
                        a_new[j+jj, i+ii] = a[jc, ic]

            elif loc == Loc.U:
                for jj in range(0, Nj):
                    for ii in range(0, Ni-Nsi):
                        a_new[j+jj, i+ii] = a[jc, ic]

                for jj in range(0, Nj):
                    ii = Ni-Nsi
                    if Ni%2 == 0:
                        a_new[j+jj, i+ii] = 0.5*(a[jc, ic] + a[jc, ic+1])
                    else:
                        a_new[j+jj, i+ii] = a[jc, ic+1]

                    for ii in range(Ni-Nsi+1, Ni):
                        a_new[j+jj, i+ii] = a[jc, ic+1]

            elif loc == Loc.V:
                for jj in range(0, Nj-Nsj):
                    for ii in range(0, Ni):
                        a_new[j+jj, i+ii] = a[jc, ic]

                jj = Nj-Nsj
                if Nj%2 == 0:
                    for ii in range(0, Ni):
                        a_new[j+jj, i+ii] = 0.5*(a[jc, ic] + a[jc+1, ic])
                else:
                    for ii in range(0, Ni):
                        a_new[j+jj, i+ii] = a[jc+1, ic]

                for jj in range(Nj-Nsj+1, Nj):
                    for ii in range(0, Ni):
                        a_new[j+jj, i+ii] = a[jc+1, ic]


## Set up the grids
xsize, ysize = 3200, 3200

dx, dy = 200, 200

xh = np.arange(-dx, xsize+dx/2, dx)
x = np.arange(-dx/2, xsize+dx, dx)

yh = np.arange(-dy, ysize+dy/2, dy)
y = np.arange(-dy/2, ysize+dy, dy)

dx_new, dy_new = dx/Ni, dy/Nj

xh_new = np.arange(-dx_new, xsize+dx_new/2, dx_new)
x_new = np.arange(-dx_new/2, xsize+dx_new, dx_new)

yh_new = np.arange(-dy_new, ysize+dy_new/2, dy_new)
y_new = np.arange(-dy_new/2, ysize+dy_new, dy_new)

u_new = np.empty((len(y_new), len(xh_new)))
v_new = np.empty((len(yh_new), len(x_new)))
s_new = np.empty((len(y_new), len(x_new)))

u_new[:] = np.nan
v_new[:] = np.nan
s_new[:] = np.nan


## U
u = np.sin(2*(2*np.pi)/xsize * xh[None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * y[:, None])
interpolate(u_new, u, 1, len(xh)-1, 1, len(y)-1, 1, 1, Loc.U)

xh_plot = np.array([ 0, *x[1:-1], xsize])
y_plot = yh[1:]

xh_new_plot = np.array([ 0, *x_new[1:-1], xsize])
y_new_plot = yh_new[1:]

plt.figure()
plt.subplot(121)
plt.pcolormesh(xh_plot, y_plot, u[1:-1, 1:])
plt.subplot(122)
plt.pcolormesh(xh_new_plot, y_new_plot, u_new[1:-1, 1:])
plt.xlim(0, xsize)
plt.ylim(0, ysize)

plt.figure()
plt.plot(xh_new, u_new[1+Nj//2, :], 'C0-^')
plt.plot(xh, u[1, :], 'k:')
plt.plot(y_new, u_new[:, 1+Ni], 'C1-^')
plt.plot(y, u[:, 2], 'k:')


## V
v = np.sin(2*(2*np.pi)/xsize * x[None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * yh[:, None])
interpolate(v_new, v, 1, len(x)-1, 1, len(yh)-1, 1, 1, Loc.V)

x_plot = xh[1:]
yh_plot = np.array([ 0, *y[1:-1], ysize])

x_new_plot = xh_new[1:]
yh_new_plot = np.array([ 0, *y_new[1:-1], ysize])

plt.figure()
plt.subplot(121)
plt.pcolormesh(x_plot, yh_plot, v[1:, 1:-1])
plt.subplot(122)
plt.pcolormesh(x_new_plot, yh_new_plot, v_new[1:, 1:-1])
plt.xlim(0, xsize)
plt.ylim(0, ysize)

plt.figure()
plt.plot(x_new, v_new[1+Nj, :], 'C0-^')
plt.plot(x, v[2, :], 'k:')
plt.plot(yh_new, v_new[:, 1+Ni//2], 'C1-^')
plt.plot(yh, v[:, 1], 'k:')


## C
s = np.sin(2*(2*np.pi)/xsize * x[None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * y[:, None])
interpolate(s_new, s, 1, len(x)-1, 1, len(y)-1, 1, 1, Loc.C)

x_plot = xh[1:]
x_new_plot = xh_new[1:]

plt.figure()
plt.subplot(121)
plt.pcolormesh(x_plot, y_plot, s[1:-1, 1:-1])
plt.subplot(122)
plt.pcolormesh(x_new_plot, y_new_plot, s_new[1:-1, 1:-1])

plt.figure()
plt.plot(x_new, s_new[1+Nj//2, :], 'C0-^')
plt.plot(x, s[1, :], 'k:')
plt.plot(y_new, s_new[:, 1+Ni//2], 'C1-^')
plt.plot(y, s[:, 1], 'k:')

plt.show()
