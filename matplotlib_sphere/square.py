import numpy as np
import matplotlib.pyplot as pl
#from mpl_toolkits.mplot3d import Axes3D
pl.ioff()

nx = 512
ny = 512

Lx = 2.e6
Ly = 2.e6

H = 1.e2
dh = 1.
g = 9.81
f = 2.e-4

dt = 10.
nt = 3000

x = np.linspace(0., Lx, nx)
y = np.linspace(0., Ly, ny)

dx = x[1] - x[0]
dy = y[1] - y[0]

dxi2 = 1./(2.*dx)
dyi2 = 1./(2.*dy)

xx, yy = np.meshgrid(x,y)

Ld = 6.e4
h0 = dh * np.exp(-( (xx - 0.5*Lx)**2 + (yy - 0.5*Ly)**2) / Ld**2 )
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

    dudt = -g * (h[2::,1:-1] - h[0:-2,1:-1])*dxi2 + f*v[1:-1, 1:-1]
    dvdt = -g * (h[1:-1,2::] - h[1:-1,0:-2])*dyi2 - f*u[1:-1, 1:-1]
    dhdt = -H * ((u[2::,1:-1] - u[0:-2,1:-1])*dxi2 + (v[1:-1,2::] - v[1:-1,0:-2])*dyi2)

    uadv = u[1:-1,1:-1]*(u[2::,1:-1] - u[0:-2,1:-1])*dxi2 + v[1:-1,1:-1]*(u[1:-1,2::] - u[1:-1,0:-2])*dyi2
    vadv = u[1:-1,1:-1]*(v[2::,1:-1] - v[0:-2,1:-1])*dxi2 + v[1:-1,1:-1]*(v[1:-1,2::] - v[1:-1,0:-2])*dyi2

    u[1:-1, 1:-1] += dt * dudt
    v[1:-1, 1:-1] += dt * dvdt
    h[1:-1, 1:-1] += dt * dhdt

    zeta = (v[2::,1:-1] - v[0:-2,1:-1])/dx + (u[1:-1,2::] - u[1:-1,0:-2])/dy
    PV = (zeta+f)/(H+h[1:-1,1:-1])
    #print("sum(PV) = {0}, nonlinear ratio = {1}".format(np.sum(PV), np.max(abs(uadv/dudt))))

    if (n%10 == 0):
        pl.figure(1)
        pl.pcolormesh(x/1000., y/1000., h[1:-1, 1:-1], cmap=pl.cm.RdYlBu_r, vmin=-0.2, vmax=0.2)
        pl.colorbar()
        #pl.quiver(xx[::10,::10], yy[::10,::10], u[::10,::10], v[::10,::10])
        pl.xlim(0, Lx/1000.)
        pl.ylim(0, Ly/1000.)
        pl.xlabel('x (km)')
        pl.ylabel('y (km)')
        pl.title('h (m) at t = {0} s'.format(1.*n*dt))
        pl.savefig('figs/{0:04d}.png'.format(n/10))
        pl.close(1)

"""
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
pl.pcolormesh(x, y, PV, cmap=pl.cm.RdYlBu)
pl.colorbar()
pl.xlim(0, Lx)
pl.ylim(0, Ly)
pl.title('PV')

#clim = np.max( abs(np.min(h[1:-1,1:-1])), np.max(h[1:-1,1:-1]) )
pl.figure()
ax = pl.subplot(111, projection='3d')
ax.plot_surface(xx/1000., yy/1000., h[1:-1,1:-1], cstride=5, rstride=5, linewidth=0)
"""
