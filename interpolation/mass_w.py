import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


N = 3


def int_lin(a_new, a, istart, iend):
    for ic in range(istart, iend):
        if N%2 == 0:
            i = (ic-istart)*N + istart + N//2

            for ii in range(1, N//2+1):
                fac = 1 - (ii/N) + 1/(2*N)
                a_new[i-ii  ] = fac*a[ic] + (1-fac)*a[ic-1]
                a_new[i+ii-1] = fac*a[ic] + (1-fac)*a[ic+1]

        else:
            i = (ic-istart)*N + istart + N//2

            a_new[i] = a[ic]
            for ii in range(1, N//2+1):
                fac = 1 - ii/N
                a_new[i-ii] = fac*a[ic] + (1-fac)*a[ic-1]
                a_new[i+ii] = fac*a[ic] + (1-fac)*a[ic+1]

def int_nn(a_new, a, istart, iend):
    for ic in range(istart, iend):
        if N%2 == 0:
            i = (ic-istart)*N + istart + N//2

            for ii in range(1, N//2+1):
                fac = 1# - (ii/N) + 1/(2*N)
                a_new[i-ii  ] = fac*a[ic] + (1-fac)*a[ic-1]
                a_new[i+ii-1] = fac*a[ic] + (1-fac)*a[ic+1]

        else:
            i = (ic-istart)*N + istart + N//2

            a_new[i] = a[ic]
            for ii in range(1, N//2+1):
                fac = 1# - ii/N
                a_new[i-ii] = fac*a[ic] + (1-fac)*a[ic-1]
                a_new[i+ii] = fac*a[ic] + (1-fac)*a[ic+1]




dx = 100
xsize = 3200

x = np.arange(-dx/2, xsize+dx, dx)
w = np.sin(8*(2*np.pi)/xsize * x) + 0.5*np.sin(5*(2*np.pi)/xsize * x)

dx_ref = dx / N
x_ref = np.arange(-dx_ref/2, xsize+dx_ref, dx_ref)
w_ref = np.sin(8*(2*np.pi)/xsize * x_ref) + 0.5*np.sin(5*(2*np.pi)/xsize * x_ref)

dx_new = dx / N

x_int = np.arange(-dx_new/2, xsize+dx_new, dx_new)

w_nn = np.empty_like(x_int)
w_nn[:] = np.nan
int_nn(w_nn, w, 1, len(x)-1)
w2_nn = w_nn**2

w_lin = np.empty_like(x_int)
w_lin[:] = np.nan
int_lin(w_lin, w, 1, len(x)-1)
w2_lin = w_lin**2

w_lin_mean = w_lin.copy()
w2_lin_mean = w2_lin.copy()
for i in range(len(w)-2):
    istart = N*i + 1
    iend = istart + N
    w_lin_mean[istart:iend] = w_lin_mean[istart:iend].mean()
    w2_lin_mean[istart:iend] = w2_lin_mean[istart:iend].mean()

w_spl = np.empty_like(x_int)
w_spl[:] = np.nan
spl = CubicSpline(x, w)
w_spl[:] = spl(x_int)
w2_spl = w_spl**2

w_spl_mean = w_spl.copy()
w2_spl_mean = w2_spl.copy()
for i in range(len(w)-2):
    istart = N*i + 1
    iend = istart + N
    w_spl_mean[istart:iend] = w_spl_mean[istart:iend].mean()
    w2_spl_mean[istart:iend] = w2_spl_mean[istart:iend].mean()


print("ref"     , w[1:-1]    .mean(), (w[1:-1]**2).mean())
print("nn"      , w_nn[1:-1] .mean(), w2_nn[1:-1] .mean())
print("lin"     , w_lin[1:-1].mean(), w2_lin[1:-1].mean())
# print("lin_mean", w_lin[1:-1].mean(), w2_lin_mean[1:-1].mean())
print("spline  ", w_spl[1:-1].mean(), w2_spl[1:-1].mean())


plt.figure()
plt.plot(x_int[1:-1], w_nn[1:-1], 'C0-', label='nn')
plt.plot(x_int[1:-1], w_lin[1:-1], 'C1-', label='lin')
# plt.plot(x_int[1:-1], w_lin_mean[1:-1])
plt.plot(x_int[1:-1], w_spl[1:-1], 'C3-', label='cubic')
# plt.plot(x_int[1:-1], w_spl_mean[1:-1], 'C3-', label='cubic')
plt.plot(x[1:-1], w[1:-1], 'ko')
plt.plot(x_ref[1:-1], w_ref[1:-1], 'k:')
plt.title('w')
plt.legend()

plt.figure()
plt.plot(x_int[1:-1], w2_nn[1:-1], 'C0-', label='nn')
plt.plot(x_int[1:-1], w2_lin[1:-1], 'C1-', label='lin')
# plt.plot(x_int[1:-1], w2_lin_mean[1:-1], 'C1-', label='lin'))
plt.plot(x_int[1:-1], w2_spl[1:-1], 'C3-', label='cubic')
# plt.plot(x_int[1:-1], w2_spl_mean[1:-1], 'C3-', label='cubic')
plt.plot(x[1:-1], w[1:-1]**2, 'ko')
plt.plot(x_ref[1:-1], w_ref[1:-1]**2, 'k:')
plt.title('w^2')
plt.legend()

plt.show()
