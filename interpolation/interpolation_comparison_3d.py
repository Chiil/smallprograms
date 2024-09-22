import matplotlib.pyplot as plt
import numpy as np
from numba import njit


Ni = 5
Nj = 3
Nk = 1


@njit
def interpolate(
        a_new, a,
        istart, iend,
        jstart, jend,
        kstart, kend,
        igc, jgc, kgc,
        T_is_u, T_is_v, T_is_w):

    Nsi = 0 if T_is_u else Ni // 2
    Nsj = 0 if T_is_v else Nj // 2
    Nsk = 0 if T_is_w else Nk // 2

    for kc in range(kstart, kend):
        for jc in range(jstart, jend):
            for ic in range(istart, iend):
                i = (ic-igc)*Ni + igc + Nsi
                j = (jc-jgc)*Nj + jgc + Nsj
                k = (kc-kgc)*Nk + kgc + Nsk

                for kk in range(-Nsk, 0):
                    for jj in range(-Nsj, 0):
                        for ii in range(-Nsi, 0):
                            fi = 1 + ii/Ni
                            fj = 1 + jj/Nj
                            fk = 1 + kk/Nk
                            a_new[k+kk, j+jj, i+ii] = (
                                    (1-fi)*(1-fj)*(1-fk)*a[kc-1, jc-1, ic-1] + fi*(1-fj)*(1-fk)*a[kc-1, jc-1, ic] + (1-fi)*fj*(1-fk)*a[kc-1, jc, ic-1] + fi*fj*(1-fk)*a[kc-1, jc, ic]
                                    + (1-fi)*(1-fj)*fk*a[kc, jc-1, ic-1] + fi*(1-fj)*fk*a[kc, jc-1, ic] + (1-fi)*fj*fk*a[kc, jc, ic-1] + fi*fj*fk*a[kc, jc, ic] )

                for kk in range(-Nsk, 0):
                    for jj in range(-Nsj, 0):
                        for ii in range(0, Ni-Nsi):
                            fi = 1 - ii/Ni
                            fj = 1 + jj/Nj
                            fk = 1 + kk/Nk
                            a_new[k+kk, j+jj, i+ii] = (
                                    fi*(1-fj)*(1-fk)*a[kc-1, jc-1, ic] + (1-fi)*(1-fj)*(1-fk)*a[kc-1, jc-1, ic+1] + fi*fj*(1-fk)*a[kc-1, jc, ic] + (1-fi)*fj*(1-fk)*a[kc-1, jc, ic+1]
                                    + fi*(1-fj)*fk*a[kc, jc-1, ic] + (1-fi)*(1-fj)*fk*a[kc, jc-1, ic+1] + fi*fj*fk*a[kc, jc, ic] + (1-fi)*fj*fk*a[kc, jc, ic+1] )

                for kk in range(-Nsk, 0):
                    for jj in range(0, Nj-Nsj):
                        for ii in range(-Nsi, 0):
                            fi = 1 + ii/Ni
                            fj = 1 - jj/Nj
                            fk = 1 + kk/Nk
                            a_new[k+kk, j+jj, i+ii] = (
                                    (1-fi)*fj*(1-fk)*a[kc-1, jc, ic-1] + fi*fj*(1-fk)*a[kc-1, jc, ic] + (1-fi)*(1-fj)*(1-fk)*a[kc-1, jc+1, ic-1] + fi*(1-fj)*(1-fk)*a[kc-1, jc+1, ic]
                                    + (1-fi)*fj*fk*a[kc, jc, ic-1] + fi*fj*fk*a[kc, jc, ic] + (1-fi)*(1-fj)*fk*a[kc, jc+1, ic-1] + fi*(1-fj)*fk*a[kc, jc+1, ic] )

                for kk in range(-Nsk, 0):
                    for jj in range(0, Nj-Nsj):
                        for ii in range(0, Ni-Nsi):
                            fi = 1 - ii/Ni
                            fj = 1 - jj/Nj
                            fk = 1 + kk/Nk
                            a_new[k+kk, j+jj, i+ii] = (
                                    fi*fj*(1-fk)*a[kc-1, jc, ic] + (1-fi)*fj*(1-fk)*a[kc-1, jc, ic+1] + fi*(1-fj)*(1-fk)*a[kc-1, jc+1, ic] + (1-fi)*(1-fj)*(1-fk)*a[kc-1, jc+1, ic+1]
                                    + fi*fj*fk*a[kc, jc, ic] + (1-fi)*fj*fk*a[kc, jc, ic+1] + fi*(1-fj)*fk*a[kc, jc+1, ic] + (1-fi)*(1-fj)*fk*a[kc, jc+1, ic+1] )

                for kk in range(0, Nk-Nsk):
                    for jj in range(-Nsj, 0):
                        for ii in range(-Nsi, 0):
                            fi = 1 + ii/Ni
                            fj = 1 + jj/Nj
                            fk = 1 - kk/Nk
                            a_new[k+kk, j+jj, i+ii] = (
                                    (1-fi)*(1-fj)*fk*a[kc, jc-1, ic-1] + fi*(1-fj)*fk*a[kc, jc-1, ic] + (1-fi)*fj*fk*a[kc, jc, ic-1] + fi*fj*fk*a[kc, jc, ic]
                                    + (1-fi)*(1-fj)*(1-fk)*a[kc+1, jc-1, ic-1] + fi*(1-fj)*(1-fk)*a[kc+1, jc-1, ic] + (1-fi)*fj*(1-fk)*a[kc+1, jc, ic-1] + fi*fj*(1-fk)*a[kc+1, jc, ic] )

                for kk in range(0, Nk-Nsk):
                    for jj in range(-Nsj, 0):
                        for ii in range(0, Ni-Nsi):
                            fi = 1 - ii/Ni
                            fj = 1 + jj/Nj
                            fk = 1 - kk/Nk
                            a_new[k+kk, j+jj, i+ii] = (
                                    fi*(1-fj)*fk*a[kc, jc-1, ic] + (1-fi)*(1-fj)*fk*a[kc, jc-1, ic+1] + fi*fj*fk*a[kc, jc, ic] + (1-fi)*fj*fk*a[kc, jc, ic+1]
                                    + fi*(1-fj)*(1-fk)*a[kc+1, jc-1, ic] + (1-fi)*(1-fj)*(1-fk)*a[kc+1, jc-1, ic+1] + fi*fj*(1-fk)*a[kc+1, jc, ic] + (1-fi)*fj*(1-fk)*a[kc+1, jc, ic+1] )

                for kk in range(0, Nk-Nsk):
                    for jj in range(0, Nj-Nsj):
                        for ii in range(-Nsi, 0):
                            fi = 1 + ii/Ni
                            fj = 1 - jj/Nj
                            fk = 1 - kk/Nk
                            a_new[k+kk, j+jj, i+ii] = (
                                    (1-fi)*fj*fk*a[kc, jc, ic-1] + fi*fj*fk*a[kc, jc, ic] + (1-fi)*(1-fj)*fk*a[kc, jc+1, ic-1] + fi*(1-fj)*fk*a[kc, jc+1, ic]
                                    + (1-fi)*fj*(1-fk)*a[kc+1, jc, ic-1] + fi*fj*(1-fk)*a[kc+1, jc, ic] + (1-fi)*(1-fj)*(1-fk)*a[kc+1, jc+1, ic-1] + fi*(1-fj)*(1-fk)*a[kc+1, jc+1, ic] )

                for kk in range(0, Nk-Nsk):
                    for jj in range(0, Nj-Nsj):
                        for ii in range(0, Ni-Nsi):
                            fi = 1 - ii/Ni
                            fj = 1 - jj/Nj
                            fk = 1 - kk/Nk
                            a_new[k+kk, j+jj, i+ii] = (
                                    + fi*fj*fk*a[kc, jc, ic] + (1-fi)*fj*fk*a[kc, jc, ic+1] + fi*(1-fj)*fk*a[kc, jc+1, ic] + (1-fi)*(1-fj)*fk*a[kc, jc+1, ic+1]
                                    + fi*fj*(1-fk)*a[kc+1, jc, ic] + (1-fi)*fj*(1-fk)*a[kc+1, jc, ic+1] + fi*(1-fj)*(1-fk)*a[kc+1, jc+1, ic] + (1-fi)*(1-fj)*(1-fk)*a[kc+1, jc+1, ic+1] )


# def interpolate_xy(
#         a_new, a,
#         istart, iend,
#         jstart, jend,
#         igc, jgc,
#         T_is_u, T_is_v, T_is_w):
# 
#     a_new_3d = a_new.reshape(a_new.shape[0], 1, a_new.shape[1])
#     a_3d = a.reshape(a.shape[0], 1, a.shape[1])
# 
#     interpolate(a_new_3d, a_3d, istart, iend, jstart, jend, 0, 1, igc, jgc, 0, T_is_u, T_is_v, T_is_w)


## Set up the grids
xsize, ysize, zsize = 6400, 6400, 3200

dx, dy, dz = 200, 200, 200

x = np.arange(-dx/2, xsize+dx, dx)
y = np.arange(-dy/2, ysize+dy, dy)
z = np.arange(-dz/2, zsize+dz, dz)

xh = np.arange(-dx, xsize+dx/2, dx)
yh = np.arange(-dy, ysize+dy/2, dy)
zh = np.arange(-dz, zsize+dz/2, dz)

dx_new, dy_new, dz_new = dx/Ni, dy/Nj, dz/Nk

x_new = np.arange(-dx_new/2, xsize+dx_new, dx_new)
y_new = np.arange(-dy_new/2, ysize+dy_new, dy_new)
z_new = np.arange(-dz_new/2, zsize+dz_new, dz_new)

xh_new = np.arange(-dx_new, xsize+dx_new/2, dx_new)
yh_new = np.arange(-dy_new, ysize+dy_new/2, dy_new)
zh_new = np.arange(-dz_new, zsize+dz_new/2, dz_new)

u_new = np.empty((len(z_new), len(y_new), len(xh_new)))
s_new = np.empty((len(z_new), len(y_new), len(x_new)))

u_new[:] = np.nan
s_new[:] = np.nan


## U
u = np.sin(2*(2*np.pi)/xsize * xh[None, None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * y[None, :, None]) * np.exp(z[:, None, None]/zsize)
interpolate(u_new, u, 1, len(xh)-1, 1, len(y)-1, 1, len(z)-1, 1, 1, 1, True, False, False)

xh_plot = np.array([ 0, *x[1:-1], xsize])
y_plot = yh[1:]

xh_new_plot = np.array([ 0, *x_new[1:-1], xsize])
y_new_plot = yh_new[1:]

plt.figure()
plt.subplot(121)
plt.pcolormesh(xh_plot, y_plot, u[1, 1:-1, 1:])
plt.subplot(122)
plt.pcolormesh(xh_new_plot, y_new_plot, u_new[1+Nk//2, 1:-1, 1:])
plt.xlim(0, xsize)
plt.ylim(0, ysize)

plt.figure()
plt.plot(xh_new, u_new[1+Nk//2, 1+Nj//2, :])
plt.plot(xh, u[1, 1, :], 'k:')
plt.plot(y_new, u_new[1+Nk//2, :, 1+Ni])
plt.plot(y, u[1, :, 2], 'k:') # We plot the second index as the first is zero.


## C
s = np.sin(2*(2*np.pi)/xsize * x[None, None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * y[None, :, None]) * np.exp(z[:, None, None]/zsize)
interpolate(s_new, s, 1, len(x)-1, 1, len(y)-1, 1, len(z)-1, 1, 1, 1, False, False, False)

x_plot = xh[1:]
x_new_plot = xh_new[1:]

plt.figure()
plt.subplot(121)
plt.pcolormesh(x_plot, y_plot, s[1, 1:-1, 1:-1])
plt.subplot(122)
plt.pcolormesh(x_new_plot, y_new_plot, s_new[1+Nk//2, 1:-1, 1:-1])

plt.figure()
plt.plot(x_new, s_new[1+Nk//2, 1+Nj//2, :])
plt.plot(x, s[1, 1, :], 'k:')
plt.plot(y_new, s_new[1+Nk//2, :, 1+Ni//2])
plt.plot(y, s[1, :, 1], 'k:')
plt.plot(z_new, s_new[:, 1+Nj+Nj//2, 1+Ni+Ni//2])
plt.plot(z, s[:, 2, 2], 'k:')


## Test a 2D interpolation
# s_bot = np.sin(2*(2*np.pi)/xsize * x[None, :]) * 0.5*np.sin(5*(2*np.pi)/ysize * y[:, None])
# 
# s_bot_new = np.empty((len(y_new), len(x_new)))
# s_bot_new[:] = np.nan
# 
# interpolate_xy(s_bot_new, s_bot, 1, len(x)-1, 1, len(y)-1, 1, 1, False, False, False)
# 
# plt.figure()
# plt.subplot(121)
# plt.pcolormesh(x_plot, y_plot, s_bot[1:-1, 1:-1])
# plt.subplot(122)
# plt.pcolormesh(x_new_plot, y_new_plot, s_bot_new[1:-1, 1:-1])
# 
# plt.figure()
# plt.plot(x_new, s_bot_new[1+N//2, :])
# plt.plot(x, s_bot[1, :], 'k:')
# plt.plot(y_new, s_bot_new[:, 1+N//2])
# plt.plot(y, s_bot[:, 1], 'k:')

plt.show()
