import numpy as np
import matplotlib.pyplot as plt


## SIMULATION SETTINGS AND GRID GENERATION
# Initializing space
n_photons = 2**18
x_range = 10

# Checking flux at these points:
dn = 0.01
arr_xh = np.arange(0, x_range+dn/2, dn) # cell edges
arr_x = np.arange(dn/2, x_range, dn) # cell centers

# Creating photon position and travel distance
arr_tau = -np.log(np.random.rand(n_photons))
arr_pos = np.random.rand(n_photons)*x_range

# RTE properties
# dI = -kext*I*dn + kext*B*dn
kext = 1.0*np.ones_like(arr_x)
# B = 1.0*np.ones_like(arr_x)
B = 1.0*arr_x/x_range


## REFERENCE SOLUTION
arr_I = np.zeros_like(arr_xh)
for i in range(1, len(arr_I)):
    arr_I[i] = arr_I[i-1] - kext[i-1]*arr_I[i-1]*dn + kext[i-1]*B[i-1]*dn


## MONTE CARLO SOLUTION
# Photon travel distance and next position
arr_dn = arr_tau / kext[0]
arr_pos_next = arr_pos + arr_dn

# Calculating power
phi_tot = kext[0] * B[0] * (x_range*1*1)
phi_per_phot = phi_tot/n_photons

# Check the flux to the cell edges
arr_flux = np.zeros_like(arr_xh)
for i, flux_point in enumerate(arr_xh):
    arr_through_cell_edge = (arr_pos < flux_point) & (arr_pos_next > flux_point)
    flux = np.sum(arr_through_cell_edge)
    arr_flux[i] = flux

# Converting MC weights to energy
arr_I_MC = arr_flux*phi_per_phot
    

# Plotting
plt.figure()
plt.plot(arr_xh, arr_I, 'r-', label=r'dI = -k$\cdot$I$\cdot$B$\cdot$dn + k$\cdot$B$\cdot$dn')
plt.plot(arr_xh, arr_I_MC, color='black', label='1D MC')
plt.plot(arr_xh, arr_I_MC - arr_I, 'k:', label='1D MC error')
plt.grid(which='major', alpha=0.5)
plt.grid(which='minor', alpha=0.2)
plt.minorticks_on()
plt.legend()
plt.title(f'1D MC longwave with {n_photons} samples')
plt.show()
