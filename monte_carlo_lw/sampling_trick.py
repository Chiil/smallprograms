import numpy as np
import matplotlib.pyplot as plt

L = 20.0
dx = 10.0
n_samples = 2**16

k_ext = 2.0

dx_bin = 0.01 * dx
L -= dx_bin # Remove 1 dx for to have the cell centered.
x = np.arange(dx_bin/2, L, dx_bin) - L/2
xh = np.arange(0, L+dx_bin/2, dx_bin) - L/2

dir_sample = np.random.randint(2, size=n_samples)
pos_sample = np.random.uniform(-dx/2, dx/2, size=n_samples)
tau_sample = -1.0 * np.log(-np.random.uniform(size=n_samples) + 1.0)

s_sample = tau_sample / k_ext

# Reference method.
pos_new = np.where(dir_sample == 0, pos_sample - s_sample, pos_sample + s_sample)
hist_ref, _ = np.histogram(pos_new, bins=xh)
hist_ref = hist_ref.astype(np.float64) / n_samples

# Weighting method.
distance_to_edge = np.where(
        dir_sample == 0,
        pos_sample + dx/2,
        dx/2 - pos_sample
)
weights = np.exp(-k_ext * distance_to_edge)
pos_new_weights = np.where(dir_sample == 0, -dx/2 - s_sample, dx/2 + s_sample)
hist_weights, _ = np.histogram(pos_new_weights, bins=xh, weights=weights)
hist_weights = hist_weights.astype(np.float64) / n_samples


plt.figure()
plt.plot(x, hist_ref, "C0+-")
plt.plot(x, hist_weights, "C1+-")
plt.show()

