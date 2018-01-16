import matplotlib.pyplot as plt
import numpy as np

s0 = 100.
a = -0.007
p = 0.

def calc_s(t):
    return (s0+p/a)*np.exp(a*t) - p/a

def calc_dsdt(t, s):
    return a*s + p

def calc_s_impl(t_start, t_end, s_start):
    dt = t_end - t_start
    return -(s_start + dt*p) / (a*dt - 1.)

t = np.arange(0, 1001, 1.)
s = calc_s(t)

tt_step = 25
ss_step = 0.125*s0
tt, ss = np.meshgrid(np.arange(0, t.max()+1, tt_step), np.arange(-.5*s0, 1.5*s0+0.001, ss_step))
dsdt_tt = np.ones(tt.shape)
dsdt_ss = calc_dsdt(tt, ss)

def trendline(t_start, t_end, s_start, method):
    dt = t_end - t_start
    if method == 'expl':
        s_end = s_start + calc_dsdt(t_start, s_start)*dt
    elif method == 'impl':
        s_end = calc_s_impl(t_start, t_end, s_start)
    elif method == 'mixed_impl':
        s_end = (1 + a*dt/2)/(1. - a*dt/2)*s_start
    elif method == 'mixed_expl':
        s_mid = s_start + calc_dsdt(t_start, s_start)*dt/2.
        s_end = s_mid + calc_dsdt(t_start + dt/2, s_mid)*dt/2.

    return np.array([t_start, t_end]), np.array([s_start, s_end])

t_end = 200

x_mixed, y_mixed = trendline(0, t_end, calc_s(0), 'mixed_impl')
x_expl, y_expl = trendline(0, 0.5*t_end, calc_s(0), 'expl')
x_impl, y_impl = trendline(0.5*t_end, t_end, y_expl[-1], 'impl')

smin, smax = 0, 1.1*s0

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
plt.plot(t[::t_end], s[::t_end], 'ro', linewidth=1.5)
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2, scale=0.06)
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.xlim(0, 1.5*t_end)
plt.ylim(smin, smax)
plt.savefig('vectorfield_7_zoom_1.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
plt.plot(t[::t_end], s[::t_end], 'ro', linewidth=1.5)
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2, scale=0.06)
plt.plot(x_mixed, y_mixed, 'k+-.')
plt.plot(x_expl, y_expl, 'k+:')
plt.plot(x_impl, y_impl, 'k+:')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.xlim(0, 1.5*t_end)
plt.ylim(smin, smax)
plt.savefig('vectorfield_7_zoom_2.pdf')
plt.close()

