import numpy as np
import matplotlib.pyplot as plt

L = 20.0
dx = 10.0
n_samples = 10_000_000

k_ext = 1.0

dx_bin = 0.01 * dx
L -= dx_bin # Remove 1 dx for to have the cell centered.
x = np.arange(dx_bin/2, L, dx_bin) - L/2
xh = np.arange(0, L+dx_bin/2, dx_bin) - L/2

dir_sample = np.random.randint(2, size=n_samples)
pos_sample = np.random.uniform(-dx/2, dx/2, size=n_samples)
tau_sample = -1.0 * np.log(-np.random.uniform(size=n_samples) + 1.0)

s_sample = tau_sample / k_ext
pos_new_sample = np.where(dir_sample == 0, pos_sample - s_sample, pos_sample + s_sample)

# Reference method.
hist_ref, _ = np.histogram(pos_new_sample, bins=xh)
# hist_ref *= dx # Show the fraction of the original cell.

# Weighting method.
distance_to_edge = np.where(
        dir_sample == 0,
        pos_sample + dx/2,
        dx/2 - pos_sample
)
weights = np.exp(-k_ext * distance_to_edge)
pos_new_weights = np.where(dir_sample == 0, -dx/2 - s_sample, dx/2 + s_sample)
hist_weights, _ = np.histogram(pos_new_weights, bins=xh)#, weights=weights)
# hist_weights *= dx # Show the fraction of the original cell.


plt.figure()
plt.plot(x, hist_ref, "C0+-")
plt.plot(x, hist_weights, "C1+-")
plt.show()

