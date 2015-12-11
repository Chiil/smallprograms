import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.basemap import Basemap, shiftgrid, addcyclic

lon = np.linspace(  0., 360., 360)
lat = np.linspace(-90.,  90., 180)

# make 2-d grid of lons, lats
lons, lats = np.meshgrid(lon, lat)

lonloc1 = 0.
latloc1 = 50.
lonloc2 = 120.
latloc2 = 20.

# angles in radians
radlons = lons*np.pi/180.
radlats = lats*np.pi/180.
radlonloc1 = lonloc1*np.pi/180.
radlatloc1 = latloc1*np.pi/180.
radlonloc2 = lonloc2*np.pi/180.
radlatloc2 = latloc2*np.pi/180.

r = 6370997.
radlondiff1 = abs(radlons-radlonloc1)
radlondiff2 = abs(radlons-radlonloc2)
d1 = r*np.arccos( np.sin(radlats)*np.sin(radlatloc1) \
                + np.cos(radlats)*np.cos(radlatloc1) * np.cos(radlondiff1) )
d2 = r*np.arccos( np.sin(radlats)*np.sin(radlatloc2) \
                + np.cos(radlats)*np.cos(radlatloc2) * np.cos(radlondiff2) )

dstd = 2.e6
h = np.exp( -d1**2 / dstd**2) - np.exp( -d2**2 / dstd**2)

# make orthographic basemap.
m = Basemap(resolution='c',projection='ortho',lat_0=60.,lon_0=100.)

# create figure, add axes
pl.close('all')
fig1 = pl.figure(figsize=(8,10))
ax = fig1.add_axes([0.1,0.1,0.8,0.8])

# compute native x,y coordinates of grid.
x, y = m(lons, lats)

# define parallels and meridians to draw.
parallels = np.arange(-80.,90,20.)
meridians = np.arange(0.,360.,20.)

# plot SLP contours.
cs = np.linspace(-1., 1., 21)
c1 = m.contour (x, y, h, cs, linewidths=0.5, colors='k', extend='both')
c2 = m.contourf(x, y, h, cs, cmap=pl.cm.RdYlBu)

# draw coastlines, parallels, meridians.
#m.drawcoastlines(linewidth=1.5)
m.drawparallels(parallels)
m.drawmeridians(meridians)

# add colorbar
cb = m.colorbar(c2, "bottom", size="5%", pad="2%")
pl.show()

#lonidx = np.abs(lon-lonloc).argmin()
#latidx = np.abs(lat-latloc).argmin()
