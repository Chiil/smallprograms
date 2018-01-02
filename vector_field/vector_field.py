import matplotlib.pyplot as plt
import numpy as np

s0 = 100.
a = -0.006
p = 0.

def calc_s(t):
    return (s0+p/a)*np.exp(a*t) - p/a

def calc_dsdt(t):
    return (s0+p/a)*a*np.exp(a*t)

def calc_s_impl(t_start, t_end, s_start):
    dt = t_end - t_start
    return -(s_start + dt*p) / (a*dt - 1.)

t = np.arange(0, 750, 1.)
s = calc_s(t)

nsteps = 15
tt, ss = np.meshgrid(np.linspace(0, t.max(), nsteps), np.linspace(-.25*s0, 1.25*s0, nsteps))
dsdt_tt = np.ones(tt.shape)
dsdt_ss = a*ss + p

def trendline(t_start, t_end, at_start=True):
    dt = t_end - t_start
    if at_start:
        s_start = calc_s(t_start)
        s_end = s_start + calc_dsdt(t_start)*dt
    else:
        s_start = calc_s(t_start)
        s_end = calc_s_impl(t_start, t_end, s_start)

    return np.array([t_start, t_end]), np.array([s_start, s_end])

t_end = 200
plt.plot(t, s, 'r-', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#aaaaaa', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.plot(*trendline(0, t_end), 'k+--')
plt.plot(*trendline(0, t_end, False), 'k+:')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.show()

