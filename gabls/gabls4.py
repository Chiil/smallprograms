import numpy as np
import pylab as pl
import netCDF4 as nc

ncfile = nc.Dataset("gabls4_les.nc","r")

z = ncfile.variables["height"][:]
theta = ncfile.variables["theta"][:]
u = ncfile.variables["u"][:]
v = ncfile.variables["v"][:]
ug = ncfile.variables["Ug"][0,:]
vg = ncfile.variables["Vg"][0,:]

ylim = 400

pl.figure()
pl.plot(theta, z, 'b-')
pl.xlim(276, 280)
pl.ylim(0  , 400)

pl.figure()
pl.plot(u , z, 'b-')
pl.plot(ug, z, 'b:')
pl.plot(v , z, 'r-')
pl.plot(vg, z, 'r:')
pl.xlim(0  ,6  )
pl.ylim(0  ,400)

pl.show()
