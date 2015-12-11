import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
pl.ion()

nx = 256
ny = 256

Lx = 1.e6
Ly = 1.e6

H = 1.e3
g = 9.81
f = 0.

dt = 5.
nt = 100

x = np.linspace(0., Lx, nx)
y = np.linspace(0., Ly, ny)

dx = x[1] - x[0]
dy = y[1] - y[0]

xx, yy = np.meshgrid(x,y)

Ld = 5.e4
h0 = 1. * np.exp(-( (xx - 0.5*Lx)**2 + (yy - 0.5*Ly)**2) / Ld**2 )
u0 = np.zeros((nx, ny))
v0 = np.zeros((nx, ny))

h = np.zeros((nx+2, ny+2))
u = np.zeros((nx+2, ny+2))
v = np.zeros((nx+2, ny+2))

h[1:-1, 1:-1] += h0

for n in range(nt):
    u[0,:] = u[-2,:]
    v[0,:] = v[-2,:]
    h[0,:] = h[-2,:]

    u[-1,:] = u[1,:]
    v[-1,:] = v[1,:]
    h[-1,:] = h[1,:]

    u[:,0] = u[:,-2]
    v[:,0] = v[:,-2]
    h[:,0] = h[:,-2]

    u[:,-1] = u[:,1]
    v[:,-1] = v[:,1]
    h[:,-1] = h[:,1]

    dudt = -g * (h[2::,1:-1] - h[0:-2,1:-1])/dx + f*v[1:-1, 1:-1]
    dvdt = -g * (h[1:-1,2::] - h[1:-1,0:-2])/dy - f*u[1:-1, 1:-1]
    dhdt = -H * ((u[2::,1:-1] - u[0:-2,1:-1])/dx + (v[1:-1,2::] - v[1:-1,0:-2])/dy)

    uadv = u[1:-1,1:-1]*(u[2::,1:-1] - u[0:-2,1:-1])/dx + v[1:-1,1:-1]*(u[1:-1,2::] - u[1:-1,0:-2])/dy
    vadv = u[1:-1,1:-1]*(v[2::,1:-1] - v[0:-2,1:-1])/dx + v[1:-1,1:-1]*(v[1:-1,2::] - v[1:-1,0:-2])/dy

    u[1:-1, 1:-1] += dt * dudt
    v[1:-1, 1:-1] += dt * dvdt
    h[1:-1, 1:-1] += dt * dhdt

    zeta = (v[2::,1:-1] - v[0:-2,1:-1])/dx + (u[1:-1,2::] - u[1:-1,0:-2])/dy
    print("sum(PV) = {0}, nonlinear ratio = {1}".format(np.sum( (zeta+f)/(H+h[1:-1,1:-1]) ), np.max(abs(uadv/dudt))))


pl.figure()
pl.pcolormesh(x, y, h[1:-1, 1:-1], cmap=pl.cm.RdYlBu)
pl.colorbar()
#pl.quiver(xx[::10,::10], yy[::10,::10], u[::10,::10], v[::10,::10])
pl.xlim(0, Lx)
pl.ylim(0, Ly)
pl.title('h')

pl.figure()
pl.pcolormesh(x, y, u[1:-1, 1:-1], cmap=pl.cm.RdYlBu)
pl.colorbar()
pl.xlim(0, Lx)
pl.ylim(0, Ly)
pl.title('u')

pl.figure()
pl.pcolormesh(x, y, v[1:-1, 1:-1], cmap=pl.cm.RdYlBu)
pl.colorbar()
pl.xlim(0, Lx)
pl.ylim(0, Ly)
pl.title('v')

pl.figure()
pl.pcolormesh(x, y, (zeta + f) / (H+h[1:-1,1:-1]), cmap=pl.cm.RdYlBu)
pl.colorbar()
pl.xlim(0, Lx)
pl.ylim(0, Ly)
pl.title('PV')

clim = np.max( abs(np.min(h[1:-1,1:-1])), np.max(h[1:-1,1:-1]) )
pl.figure()
ax = pl.subplot(111, projection='3d')
ax.plot_surface(xx/1000., yy/1000., h[1:-1,1:-1], cstride=5, rstride=5, linewidth=0, cmap=pl.cm.RdYlBu_r, vmin=-clim, vmax=clim)

