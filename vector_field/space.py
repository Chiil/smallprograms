import matplotlib.pyplot as plt
import numpy as np

nx = 1000
x = np.arange(0, 100, 100/nx)

n = 2
h = np.sin(n*2*np.pi/100*x)
dhdx = n*2*np.pi/100*np.cos(n*2*np.pi/100*x)
d2hdx2 = -(n*2*np.pi/100)**2*np.sin(n*2*np.pi/100*x)

def diff_c(nx):
    x = np.arange(0, 100, 100/nx)
    h = np.sin(n*2*np.pi/100*x)
    dhdx = np.zeros(h.shape)
    dhdx[0] = np.nan
    dhdx[-1] = np.nan
    dhdx[1:-1] = (h[2:] - h[0:-2]) / (x[2:] - x[0:-2])
    return x, h, dhdx

def diff_f(nx):
    x = np.arange(0, 100, 100/nx)
    h = np.sin(n*2*np.pi/100*x)
    dhdx = np.zeros(h.shape)
    dhdx[-1] = np.nan
    dhdx[0:-1] = (h[1:] - h[0:-1]) / (x[1:] - x[0:-1])
    return x, h, dhdx

def diff_b(nx):
    x = np.arange(0, 100, 100/nx)
    h = np.sin(n*2*np.pi/100*x)
    dhdx = np.zeros(h.shape)
    dhdx[0] = np.nan
    dhdx[1:] = (h[1:] - h[0:-1]) / (x[1:] - x[0:-1])
    return x, h, dhdx

def diff_2(nx):
    x = np.arange(0, 100, 100/nx)
    h = np.sin(n*2*np.pi/100*x)
    d2hdx2 = np.zeros(h.shape)
    d2hdx2[ 0] = np.nan
    d2hdx2[-1] = np.nan
    dx = x[1] - x[0]
    d2hdx2[1:-1] = (h[2:] - 2*h[1:-1] + h[0:-2]) / dx**2
    return d2hdx2

x_10_f, h_10_f, dhdx_10_f = diff_f(10)
x_10_b, h_10_b, dhdx_10_b = diff_b(10)
x_10_c, h_10_c, dhdx_10_c = diff_c(10)

x_20_f, h_20_f, dhdx_20_f = diff_f(20)
x_20_b, h_20_b, dhdx_20_b = diff_b(20)
x_20_c, h_20_c, dhdx_20_c = diff_c(20)

x_40_f, h_40_f, dhdx_40_f = diff_f(40)
x_40_b, h_40_b, dhdx_40_b = diff_b(40)
x_40_c, h_40_c, dhdx_40_c = diff_c(40)

d2hdx2_10 = diff_2(10)
d2hdx2_20 = diff_2(20)
d2hdx2_40 = diff_2(40)

plt.figure()
plt.subplot(211)
plt.plot(x, h, 'C0-')
plt.plot(x_10_f, h_10_f, 'C0o')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.subplot(212)
plt.plot(x, dhdx, 'k:')
plt.xlabel('x (m)')
plt.ylabel('dsdx (1/m)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('spatial_1.pdf')

plt.figure()
plt.subplot(211)
plt.plot(x, h, 'C0-')
plt.plot(x_10_f, h_10_f, 'C0o')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.subplot(212)
plt.plot(x_10_f, dhdx_10_f, label='f')
plt.plot(x_10_b, dhdx_10_b, label='b')
plt.plot(x_10_c, dhdx_10_c, label='c')
plt.plot(x, dhdx, 'k:')
plt.xlabel('x (m)')
plt.ylabel('dsdx (1/m)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('spatial_2.pdf')

plt.figure()
plt.subplot(211)
plt.plot(x, h, 'C0-')
plt.plot(x_20_f, h_20_f, 'C0o')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.subplot(212)
plt.plot(x_20_f, dhdx_20_f, label='f')
plt.plot(x_20_b, dhdx_20_b, label='b')
plt.plot(x_20_c, dhdx_20_c, label='c')
plt.plot(x, dhdx, 'k:')
plt.xlabel('x (m)')
plt.ylabel('dsdx (1/m)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('spatial_3.pdf')

plt.figure()
plt.subplot(211)
plt.plot(x, h, 'C0-')
plt.plot(x_40_f, h_40_f, 'C0o')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.subplot(212)
plt.plot(x_40_f, dhdx_40_f, label='f')
plt.plot(x_40_b, dhdx_40_b, label='b')
plt.plot(x_40_c, dhdx_40_c, label='c')
plt.plot(x, dhdx, 'k:')
plt.xlabel('x (m)')
plt.ylabel('dsdx (1/m)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('spatial_4.pdf')

plt.figure()
plt.subplot(211)
plt.plot(x_10_f, dhdx_10_f)
plt.plot(x_20_f, dhdx_20_f)
plt.plot(x_40_f, dhdx_40_f)
plt.plot(x, dhdx, 'k:')
plt.xlabel('x (m)')
plt.ylabel('dsdx_f (1/m)')
plt.subplot(212)
plt.plot(x_10_f, dhdx_10_c)
plt.plot(x_20_f, dhdx_20_c)
plt.plot(x_40_f, dhdx_40_c)
plt.plot(x, dhdx, 'k:')
plt.xlabel('x (m)')
plt.ylabel('dsdx_c (1/m)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('spatial_5.pdf')

plt.figure()
plt.subplot(211)
plt.plot(x, h, 'C0-')
plt.plot(x_10_f, h_10_f, 'C0o')
plt.xlabel('x (m)')
plt.ylabel('s (-)')
plt.subplot(212)
plt.plot(x_10_f, d2hdx2_10)
plt.plot(x_20_f, d2hdx2_20)
plt.plot(x_40_f, d2hdx2_40)
plt.plot(x, d2hdx2, 'k:')
plt.xlabel('x (m)')
plt.ylabel('d2sdx2 (1/m2)')
plt.legend(loc=0, frameon=False)
plt.tight_layout()
plt.savefig('spatial_6.pdf')
plt.show()

