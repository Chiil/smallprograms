import matplotlib.pyplot as plt
import numpy as np
from numba import njit

@njit
def nn(a, a_new):
    for n in range(1, len(a)-1):
        i = n
        ii = 3*(n-1)+1
        a_new[ii-1] = a[i]
        a_new[ii  ] = a[i]
        a_new[ii+1] = a[i]

@njit
def linear(a, a_new):
    for n in range(1, len(a)-1):
        i = n
        ii = 3*(n-1)+1
        a_new[ii-1] = 1/3*a[i-1] + 2/3*a[i]
        a_new[ii  ] = a[i]
        a_new[ii+1] = 2/3*a[i] + 1/3*a[i+1]

@njit
def linear_cons(a, a_new):
    for n in range(1, len(a)-1):
        i = n
        ii = 3*(n-1)+1

        a_new_m = 1/3*a[i-1] + 2/3*a[i]
        a_new_o = a[i]
        a_new_p = 2/3*a[i] + 1/3*a[i+1]

        a_factor = 3*a[i] / (a_new_m + a_new_o + a_new_p)

        a_new[ii-1] = a_factor * a_new_m
        a_new[ii  ] = a_factor * a_new_o
        a_new[ii+1] = a_factor * a_new_p


# Interpolation functions
dx = 100
xsize = 3200

x = np.arange(-dx/2, xsize+dx, dx)
w = np.sin(2*(2*np.pi)/xsize * x) + 0.5*np.sin(5*(2*np.pi)/xsize * x)

dx_ref = dx/3
x_ref = np.arange(dx_ref/2, xsize, dx_ref)
w_ref = np.sin(2*(2*np.pi)/xsize * x_ref) + 0.5*np.sin(5*(2*np.pi)/xsize * x_ref)

w_nn = np.zeros_like(w_ref)
w_lin = np.zeros_like(w_ref)
w_lin_cons = np.zeros_like(w_ref)

nn(w, w_nn)
linear(w, w_lin)
linear_cons(w, w_lin_cons)

print(f"w = {w[1:-1].sum()}")
print(f"w_ref = {w_ref.sum()}")
print(f"w_nn = {w_nn .sum()}")
print(f"w_lin = {w_lin.sum()}")
print(f"w_lin_cons = {w_lin_cons.sum()}")

plt.figure()
plt.plot(x_ref, w_nn, label='nn')
plt.plot(x_ref, w_lin, label='linear')
plt.plot(x_ref, w_lin_cons, label='linear_cons')
plt.plot(x_ref, w_ref, 'k:')
plt.legend()

plt.show()
