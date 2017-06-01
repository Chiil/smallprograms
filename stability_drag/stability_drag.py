import numpy as np
import matplotlib.pyplot as plt

u0 = 1.
dt = 1.
nt = 10
cd = 1.1

t = np.linspace(0, nt*dt, nt+1)
u = np.zeros(nt+1)

u[0] = u0
for i in range(1,nt+1):
    u[i] = u[i-1] - dt*cd*u[i-1]**2

plt.figure()
plt.plot(t, u)
plt.show()
