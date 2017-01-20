import numpy as np
import pylab as pl
pl.ion()

def rhs(r, x):
    return r * x * (1.-x)

def logistic_map(x0, r, nt):
    x = np.zeros((nt, r.size))
    x[0] = x0

    for n in range(1, nt):
        x[n,:] = rhs(r, x[n-1,:])

    return x

def cobweb(x0, r, nt):
    x = np.zeros(nt*2)
    y = np.zeros(nt*2)
    x[0] = x0
    y[0] = 0.
    x[1] = x0
    y[1] = rhs(r, x[1])
    for n in range(1, nt):
        n2 = n*2
        x[n2]   = y[n2-1]
        y[n2]   = x[n2]
        x[n2+1] = x[n2]
        y[n2+1] = rhs(r, x[n2+1])

    return x, y

x0 = 0.1

# BIFURCATION DIAGRAM
nt = 2000
r = np.linspace(2.8, 4., nt)
x = logistic_map(x0, r, nt)

pl.close('all')
pl.figure()
for n in range(nt-400,nt):
    pl.plot(r, x[n,:], 'k.', markersize=0.06, alpha=0.4)
pl.xlabel('r')
pl.ylabel('x')
pl.xlim(r[0], r[-1])
pl.savefig('bifurcation.png', dpi=300)


# COBWEB DIAGRAMS
ntc = 1000
xx = np.linspace(0., 1., 1000)

rs = [2.8, 3.2, 3.5, 3.83, 4.]
for rr in rs:
    xr, yr = cobweb(x0, rr, ntc)
    yy = rhs(rr, xx)

    pl.figure()
    pl.plot(xx, yy, 'k-')
    pl.plot(xx, xx, 'k-')
    pl.plot(xr[0:100], yr[0:100], color='#cccccc')
    pl.plot(xr[ntc//2+1:: ], yr[ntc//2+1:: ], 'r-')
    pl.plot(xr[ntc//2+1::2], yr[ntc//2+1::2], 'ro')
    pl.xlabel('xn'  )
    pl.ylabel('xn+1')
    pl.title('r = {0}'.format(rr))
    pl.savefig('cobweb{0}.png'.format(int(rr*10)), dpi=300)
