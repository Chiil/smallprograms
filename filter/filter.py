import numpy as np
import pylab as pl
pl.ion()

def filter_profile(prof, n):
    new_prof = np.copy(prof)
    for i in range(n):
        new_prof[1:-1] = (new_prof[0:-2] + new_prof[2:]) / 2.
    return new_prof

dz = 50
z = np.arange(dz/2, 10000, dz)
theta = np.zeros(z.size)
theta0 = 300.
dthetadz = 0.005
theta[0] = theta0 + 0.5*dz*dthetadz

for k in range(1, z.size):
    if (z[k] < 5000):
        theta[k] = theta[k-1] + dthetadz*dz
    else:
        theta[k] = theta[k-1] + 2.*dthetadz*dz

pl.plot(theta,z)
pl.plot(filter_profile(theta, 1000),z)

