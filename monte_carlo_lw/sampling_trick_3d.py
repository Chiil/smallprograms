import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numba import njit
import cmaps

itot = 15
jtot = 15
ktot = 15

dx = 40.0
dy = 20.0
dz = 10.0

Lx = itot*dx
Ly = jtot*dy
Lz = ktot*dz

n_samples = 2**16

k_ext = 5.0

# Generate a grid.
x = np.arange(dx/2, Lx, dx) - Lx/2
xh = np.arange(0, Lx+dx/2, dx) - Lx/2

y = np.arange(dy/2, Ly, dy) - Ly/2
yh = np.arange(0, Ly+dy/2, dy) - Ly/2

z = np.arange(dz/2, Lz, dz) - Lz/2
zh = np.arange(0, Lz+dz/2, dz) - Lz/2

theta_sample = np.random.uniform(-np.pi, np.pi, size=n_samples)
phi_sample = np.random.uniform(0.0, 2.0*np.pi, size=n_samples)
pos_sample_x = np.random.uniform(-dx/2, dx/2, size=n_samples)
pos_sample_y = np.random.uniform(-dy/2, dy/2, size=n_samples)
pos_sample_z = np.random.uniform(-dz/2, dz/2, size=n_samples)
pos_sample = np.array([pos_sample_z, pos_sample_y, pos_sample_x]).T
tau_sample = -1.0 * np.log(-np.random.uniform(size=n_samples) + 1.0)
dn_sample = tau_sample / k_ext

print(pos_sample.shape)

@njit
def count_abs(pos):
    abs_sample = np.zeros((len(z), len(y), len(x)))
    
    for i in range(n_samples):
        k = int((pos_sample[i, 0] + Lz/2) / dz) 
        j = int((pos_sample[i, 1] + Ly/2) / dy)
        i = int((pos_sample[i, 2] + Lx/2) / dx)
        abs_sample[k, j, i] += 1

    return abs_sample / n_samples

# Reference method.
abs_ref = count_abs(pos_sample)
print(abs_ref[ktot//2, jtot//2, itot//2])

plt.figure(figsize=(12, 4), constrained_layout=True)
plt.subplot(131)
plt.pcolormesh(xh, yh, abs_ref[ktot//2, :, :], cmap=cmaps.WhiteBlueGreenYellowRed)
plt.gca().add_patch(Rectangle((-dx/2, -dy/2), dx, dy, linestyle=':', fill=False))
plt.subplot(132)
plt.pcolormesh(xh, zh, abs_ref[:, jtot//2, :], cmap=cmaps.WhiteBlueGreenYellowRed)
plt.gca().add_patch(Rectangle((-dx/2, -dz/2), dx, dz, linestyle=':', fill=False))
plt.subplot(133)
plt.pcolormesh(yh, zh, abs_ref[:, :, itot//2], cmap=cmaps.WhiteBlueGreenYellowRed)
plt.gca().add_patch(Rectangle((-dy/2, -dz/2), dy, dz, linestyle=':', fill=False))
plt.show()
