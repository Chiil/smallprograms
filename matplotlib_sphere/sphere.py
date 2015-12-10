import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a sphere.
r = 1.
phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]


x = (r + 0.2*np.exp( -(phi)**2. / 0.04) ) * np.sin(phi) * np.cos(theta)
y = (r + 0.2*np.exp( -(phi)**2. / 0.04) ) * np.sin(phi) * np.sin(theta)
z = (r + 0.2*np.exp( -(phi)**2. / 0.04) ) * np.cos(phi)

# Set colours and render.
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=1., linewidth=0.)
ax.plot_surface(x, y, z,  rstride=1, cstride=1, cmap=pl.cm.gist_earth, vmin=1., vmax=1.2, linewidth=0.)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_aspect("equal")
pl.tight_layout()
pl.show()

