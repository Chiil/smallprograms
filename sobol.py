from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np

n = 8192

sampler = qmc.Sobol(d=2, scramble=True, seed=4)
s1 = sampler.random(n)

plt.figure()
plt.plot(s1[:,0], s1[:,1], 'C0.')
plt.xlim(0, 1)
plt.ylim(0, 1)


s2 = s1.copy()

nx = 8
ny = 8
nbox = nx*ny
nperbox = n // nbox

for ibox in range(0, nbox):
    xoff = (1/nx)*(ibox // nx)
    yoff = (1/ny)*(ibox % nx)
    s2[ibox*nperbox:(ibox+1)*nperbox, 0] /= nx
    s2[ibox*nperbox:(ibox+1)*nperbox, 1] /= ny
    s2[ibox*nperbox:(ibox+1)*nperbox, 0] += xoff
    s2[ibox*nperbox:(ibox+1)*nperbox, 1] += yoff


plt.figure()
plt.plot(s2[:,0], s2[:,1], 'C1.')
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.figure()
for ibox in range(0, nbox):
    plt.scatter(s2[ibox*nperbox:(ibox+1)*nperbox, 0], s2[ibox*nperbox:(ibox+1)*nperbox, 1], marker='.')
plt.xlim(0, 1)
plt.ylim(0, 1)


plt.show()
