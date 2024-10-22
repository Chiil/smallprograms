import matplotlib.pyplot as plt
import numpy as np
from numba import njit

Ni = 8
Nj = 8

# @njit
def interpolate(a_new, a, istart, iend, jstart, jend, igc, jgc, T_is_u, T_is_v):
    Nsi = 0 if T_is_u else Ni // 2
    Nsj = 0 if T_is_v else Nj // 2

    for jc in range(jstart, jend):
        for ic in range(istart, iend):
            i = (ic-igc)*Ni + igc + Nsi
            j = (jc-jgc)*Nj + jgc + Nsj

            if Ni%2 == 0:
                for jj in range(-Nsj, 0):
                    for ii in range(-Nsi, 0):
                        fi = 1 + ii/Ni + 1/(2*Ni)
                        fj = 1 + jj/Nj + 1/(2*Nj)
                        a_new[j+jj, i+ii] = (1-fi)*(1-fj)*a[jc-1, ic-1] + fi*(1-fj)*a[jc-1, ic] + (1-fi)*fj*a[jc, ic-1] + fi*fj*a[jc, ic]

                for jj in range(-Nsj, 0):
                    for ii in range(1, Ni-Nsi+1):
                        fi = 1 - ii/Ni + 1/(2*Ni)
                        fj = 1 + jj/Nj + 1/(2*Nj)
                        a_new[j+jj, i+ii-1] = fi*(1-fj)*a[jc-1, ic] + (1-fi)*(1-fj)*a[jc-1, ic+1] + fi*fj*a[jc, ic] + (1-fi)*fj*a[jc, ic+1]

                for jj in range(1, Nj-Nsj+1):
                    for ii in range(-Nsi, 0):
                        fi = 1 + ii/Ni + 1/(2*Ni)
                        fj = 1 - jj/Nj + 1/(2*Nj)
                        a_new[j+jj-1, i+ii] = (1-fi)*fj*a[jc, ic-1] + fi*fj*a[jc, ic] + (1-fi)*(1-fj)*a[jc+1, ic-1] + fi*(1-fj)*a[jc+1, ic]

                for jj in range(1, Nj-Nsj+1):
                    for ii in range(1, Ni-Nsi+1):
                        fi = 1 - ii/Ni + 1/(2*Ni)
                        fj = 1 - jj/Nj + 1/(2*Nj)
                        a_new[j+jj-1, i+ii-1] = fi*fj*a[jc, ic] + (1-fi)*fj*a[jc, ic+1] + fi*(1-fj)*a[jc+1, ic] + (1-fi)*(1-fj)*a[jc+1, ic+1]

            else:
                for jj in range(-Nsj, 0):
                    for ii in range(-Nsi, 0):
                        fi = 1 + ii/Ni
                        fj = 1 + jj/Nj
                        a_new[j+jj, i+ii] = (1-fi)*(1-fj)*a[jc-1, ic-1] + fi*(1-fj)*a[jc-1, ic] + (1-fi)*fj*a[jc, ic-1] + fi*fj*a[jc, ic]

                for jj in range(-Nsj, 0):
                    for ii in range(0, Ni-Nsi+1):
                        fi = 1 - ii/Ni
                        fj = 1 + jj/Nj
                        a_new[j+jj, i+ii] = fi*(1-fj)*a[jc-1, ic] + (1-fi)*(1-fj)*a[jc-1, ic+1] + fi*fj*a[jc, ic] + (1-fi)*fj*a[jc, ic+1]

                for jj in range(0, Nj-Nsj):
                    for ii in range(-Nsi, 0):
                        fi = 1 + ii/Ni
                        fj = 1 - jj/Nj
                        a_new[j+jj, i+ii] = (1-fi)*fj*a[jc, ic-1] + fi*fj*a[jc, ic] + (1-fi)*(1-fj)*a[jc+1, ic-1] + fi*(1-fj)*a[jc+1, ic]

                for jj in range(0, Nj-Nsj+1):
                    for ii in range(0, Ni-Nsi+1):
                        fi = 1 - ii/Ni
                        fj = 1 - jj/Nj
                        a_new[j+jj, i+ii] = fi*fj*a[jc, ic] + (1-fi)*fj*a[jc, ic+1] + fi*(1-fj)*a[jc+1, ic] + (1-fi)*(1-fj)*a[jc+1, ic+1]


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
s_new = np.empty((len(y_new), len(x_new)))

u_new[:] = np.nan
s_new[:] = np.nan


## U
u = np.sin(2*(2*np.pi)/xsize * xh[None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * y[:, None])
interpolate(u_new, u, 1, len(xh)-1, 1, len(y)-1, 1, 1, True, False)

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
plt.plot(xh_new, u_new[1+Nj//2, :])
plt.plot(xh, u[1, :], 'k:')
plt.plot(y_new, u_new[:, 1+Ni])
plt.plot(y, u[:, 2], 'k:')


## C
s = np.sin(2*(2*np.pi)/xsize * x[None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * y[:, None])
interpolate(s_new, s, 1, len(x)-1, 1, len(y)-1, 1, 1, False, False)

x_plot = xh[1:]
x_new_plot = xh_new[1:]

plt.figure()
plt.subplot(121)
plt.pcolormesh(x_plot, y_plot, s[1:-1, 1:-1])
plt.subplot(122)
plt.pcolormesh(x_new_plot, y_new_plot, s_new[1:-1, 1:-1])

plt.figure()
plt.plot(x_new, s_new[1+Nj//2, :])
plt.plot(x, s[1, :], 'k:')
plt.plot(y_new, s_new[:, 1+Ni//2])
plt.plot(y, s[:, 1], 'k:')

plt.show()
