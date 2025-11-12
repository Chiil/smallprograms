import numpy as np
import matplotlib.pyplot as plt

L = 100.0
dx = 5.0
n_samples = 1_000_000

k_ext = 0.25

L -= dx # Remove 1 dx for to have the cell centered.
x = np.arange(dx/2, L, dx) - L/2
xh = np.arange(0, L+dx/2, dx) - L/2

dir_sample = np.random.randint(2, size=n_samples)
pos_sample = np.random.uniform(-dx/2, dx/2, size=n_samples)
tau_sample = -1.0 * np.log(-np.random.uniform(size=n_samples) + 1.0)

s_sample = tau_sample / k_ext
pos_new_sample = np.where(dir_sample == 0, pos_sample - s_sample, pos_sample + s_sample)

hist, _ = np.histogram(pos_new_sample, bins=xh, density=True)
hist *= dx # Show the fraction of the original cell.

plt.figure()
plt.plot(x, hist, "C0o-")
plt.show()

