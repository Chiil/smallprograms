import numpy as np
import matplotlib.pyplot as plt

# Initializing space
Nphots         = int(5e4)
x_range        = 10 # from x = 0 to x = 1000

# Checking flux at these points:
dn = 0.01
arr_flux_point = np.arange(0, x_range + dn, dn)

# Creating photon position and travel distance
arr_tau = -np.log(np.random.rand(Nphots))
arr_pos = np.random.rand(Nphots)*x_range

# RTE properties
# dI = -kext*I*dn + kext*B*dn
kext = 1
B    = 1

# Creating analytical solution
arr_I = np.zeros(len(arr_flux_point))
for i in range(1, len(arr_I)):
    arr_I[i] = arr_I[i-1] + -kext*arr_I[i-1]*dn + kext*B*dn

# Photon travel distance and next position
arr_dn       = arr_tau / kext
arr_pos_next = arr_pos + arr_dn


# Calculating power
phi_tot = kext * B * (x_range*1*1)
phi_per_phot = phi_tot/Nphots

# Checking flux at flux points
arr_FW = np.zeros(len(arr_flux_point))
for i, flux_point in enumerate(arr_flux_point):
    arr_going_through_F = (arr_pos < flux_point) & (arr_pos_next > flux_point)
    FW = np.sum(arr_going_through_F)
    arr_FW[i] = FW

# Converting MC weights to energy
arr_F = arr_FW*phi_per_phot
    

# Plotting
plt.figure()
plt.plot(arr_flux_point, arr_I, color='red', 
         label=r'dI = -k$\cdot$I$\cdot$B$\cdot$dn + k$\cdot$B$\cdot$dn', 
         linewidth=3, linestyle='dashed')
plt.plot(arr_flux_point, arr_F, color='black', label='1D Monte Carlo')
plt.plot(arr_flux_point, arr_F - arr_I, 'k:', label='1D MC error')
plt.grid(which='major', alpha=0.5)
plt.grid(which='minor', alpha=0.2)
plt.minorticks_on()
plt.legend()
plt.title('1D Monte Carlo vs Radiation transfer EQ')
plt.show()
