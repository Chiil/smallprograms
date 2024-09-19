import matplotlib.pyplot as plt
import numpy as np

# Interpolation functions


dx = 100
xsize = 3200

x = np.arange(-dx/2, xsize+dx, dx)
w = np.sin(2*(2*np.pi)/xsize * x)

dx_ref = dx/3
x_ref = np.arange(dx_ref/2, xsize, dx_ref)
w_ref = np.sin(2*(2*np.pi)/xsize * x_ref)


plt.figure()
plt.plot(x[1:-1], w[1:-1])
plt.plot(x_ref, w_ref, 'k:')

plt.show()
