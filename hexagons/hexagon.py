import numpy as np
import pylab as pl
from scipy import interpolate

nx = 100
Lx = 1.
dx = Lx / nx
dy = (.75)**.5 * dx

ny = int(Lx / dy)
Ly = Lx - Lx % dy

print(nx, ny)

points = []

for j in range(ny+1):
    y = j*dy
    j_odd = j % 2
    for i in range(nx+1-j_odd):
        x = (i + j_odd/2)*dx
        points.append((x,y))

points = np.array(points)
x = points[:,0]
y = points[:,1]

xm = 0.5
ym = 0.5
sigma = 0.05

z = np.exp( - ((x-xm)**2) / (2.*sigma**2) ) \
  * np.exp( - ((y-ym)**2) / (2.*sigma**2) )

x_plot, y_plot = np.meshgrid(np.linspace(0, Lx, 1000), np.linspace(0, Ly, 1000))
#z_plot = interpolate.griddata(points, z, (x_plot, y_plot), method='nearest')
z_plot = interpolate.griddata(points, z, (x_plot, y_plot), method='cubic')

pl.figure()
pl.subplot(111, aspect='equal')
pl.plot(x, y, 'k.', markersize=0.5)
pl.pcolormesh(x_plot, y_plot, z_plot, vmin=0., vmax=1.)
pl.colorbar()
pl.xlim(0, Lx)
pl.ylim(0, Ly)
pl.show()
