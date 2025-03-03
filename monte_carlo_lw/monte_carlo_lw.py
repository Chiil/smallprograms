import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import qmc


## SIMULATION SETTINGS AND GRID GENERATION
# Initializing space
n_photons_pow = 16
n_photons = 2**n_photons_pow
x_range = 10
do_quasi_random = True

# Checking flux at these points:
dn = 0.01
arr_xh = np.arange(0, x_range+dn/2, dn) # cell edges
arr_x = np.arange(dn/2, x_range, dn) # cell centers

# RTE properties: dI = -kext*I*dn + kext*B*dn

def calc_kext(x):
    # return 1.0
    # return 0.1 + 1.0/x_range*x
    return 2.0 + np.sin(2.0*np.pi*x/x_range)

def calc_kext_int(x):
    # return 1.0*x
    # return 0.1*x + 0.5/x_range*x**2
    return 2.0*x - x_range/(2.0*np.pi)*np.cos(2.0*np.pi*x/x_range)

def calc_pos_next(pos, tau):
    kext_int_start = calc_kext_int(pos)
    pos_next = fsolve(lambda x: calc_kext_int(x) - kext_int_start - tau, 0)
    return pos_next[0]

def calc_B(x):
    # return 1.0
    # return 1.0*x/x_range
    return 1.0*(x/x_range)**2
    # return 1.0 + np.sin(2.0*np.pi*x/x_range)

kext = np.array([ calc_kext(x) for x in arr_x ])
B = np.array([ calc_B(x) for x in arr_x ])
phi_tot = dn * np.sum(kext[:] * B[:])


## REFERENCE SOLUTION
arr_I = np.zeros_like(arr_xh)
for i in range(1, len(arr_I)):
    arr_I[i] = arr_I[i-1] - kext[i-1]*arr_I[i-1]*dn + kext[i-1]*B[i-1]*dn


## MONTE CARLO SOLUTION
# Creating photon position and travel distance
arr_tau = - np.log(np.random.rand(n_photons))
if do_quasi_random:
    sampler = qmc.Sobol(d=1, scramble=True)
    arr_pos = sampler.random_base2(m=n_photons_pow).flatten()*x_range
else:
    arr_pos = np.random.rand(n_photons)*x_range

arr_pos_next = np.array( [ calc_pos_next(pos, tau) for pos, tau in np.c_[arr_pos, arr_tau] ] )

# B at pos location.
arr_phi = np.array([ calc_kext(x)*calc_B(x) for x in arr_pos ])
arr_phi *= phi_tot / arr_phi.sum()

# Check the flux to the cell edges
arr_I_MC = np.zeros_like(arr_xh)
for i, flux_point in enumerate(arr_xh):
    phi_through_cell_edge = arr_phi[(arr_pos < flux_point) & (arr_pos_next > flux_point)]
    arr_I_MC[i] = np.sum(phi_through_cell_edge)

arr_I_error = dn*np.sum((arr_I_MC - arr_I)**2)
print('L2 error:', arr_I_error)


## PLOTTING COMPARISON
plt.figure()
plt.plot(arr_xh, arr_I, 'r-', label=r'dI = -k$\cdot$I$\cdot$B$\cdot$dn + k$\cdot$B$\cdot$dn')
plt.plot(arr_xh, arr_I_MC, color='black', label='1D MC')
plt.plot(arr_xh, arr_I_MC - arr_I, 'k:', label='1D MC error')
plt.grid(which='major', alpha=0.5)
plt.grid(which='minor', alpha=0.2)
plt.minorticks_on()
plt.legend()
plt.title(f'1D MC longwave with {n_photons:,} samples, do_quasi_random = {do_quasi_random}')
plt.show()

