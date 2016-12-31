import numpy as np
import matplotlib.pyplot as pl

n = 100
p0 = 1e5
g = 9.81
dp = 5000.
L = 1e6
H = 1e4
dpdx = dp/L

x = np.linspace(0., L, n)
z = np.linspace(0., H, n)

p = p0 - g*z[:,np.newaxis] + dpdx*x[np.newaxis,:]

pl.figure()
pl.contourf(x, z, p, 20)
pl.grid()
pl.colorbar()
pl.xlabel('x (m)')
pl.ylabel('z (m)')
pl.savefig('z_coords.png', dpi=150)

xx, zz = np.meshgrid(x, z)

pl.figure()
pl.contourf(xx, p, zz, 20)
pl.colorbar()
pl.grid()
pl.gca().invert_yaxis()
pl.xlabel('x (m)')
pl.ylabel('p (Pa)')

pl.savefig('p_coords.png', dpi=150)

