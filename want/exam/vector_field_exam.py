import matplotlib.pyplot as plt
import numpy as np

plot_shift = 2.5

def calc_s(t):
    return 1./(1. + np.exp(-t))

def calc_s_impl(t_start, t_end, s_start):
    dt = t_end - t_start
    return ((4*dt*s_start + dt**2 - 2*dt + 1)**.5 + dt - 1)/(2*dt)

def calc_dsdt(t, s):
    return s*(1.-s)

def trendline(t_start, t_end, s_start, method):
    dt = t_end - t_start
    if method == 'expl':
        s_end = s_start + calc_dsdt(t_start, s_start)*dt
    elif method == 'impl':
        s_end = calc_s_impl(t_start, t_end, s_start)
    elif method == 'mixed_impl':
        s_mid = s_start + calc_dsdt(t_start, s_start)*dt/2.
        s_end = calc_s_impl(t_start + dt/2, t_end, s_mid)

    return np.array([t_start, t_end]), np.array([s_start, s_end])

t_step = 4.*np.arange(5) - plot_shift
x_expl_1, y_expl_1 = trendline(t_step[0], t_step[1], calc_s(t_step[0]), 'expl')
x_expl_2, y_expl_2 = trendline(t_step[1], t_step[2], y_expl_1[-1], 'expl')
x_expl_3, y_expl_3 = trendline(t_step[2], t_step[3], y_expl_2[-1], 'expl')
x_expl_4, y_expl_4 = trendline(t_step[3], t_step[4], y_expl_3[-1], 'expl')

x_impl_1, y_impl_1 = trendline(t_step[0], t_step[1], calc_s(t_step[0]), 'impl')
x_impl_2, y_impl_2 = trendline(t_step[1], t_step[2], y_impl_1[-1], 'impl')
x_impl_3, y_impl_3 = trendline(t_step[2], t_step[3], y_impl_2[-1], 'impl')
x_impl_4, y_impl_4 = trendline(t_step[3], t_step[4], y_impl_3[-1], 'impl')

x_mixed_1, y_mixed_1 = trendline(t_step[0], t_step[1], calc_s(t_step[0]), 'mixed_impl')
x_mixed_2, y_mixed_2 = trendline(t_step[1], t_step[2], y_mixed_1[-1], 'mixed_impl')
x_mixed_3, y_mixed_3 = trendline(t_step[2], t_step[3], y_mixed_2[-1], 'mixed_impl')
x_mixed_4, y_mixed_4 = trendline(t_step[3], t_step[4], y_mixed_3[-1], 'mixed_impl')

t = np.arange(-10., 10.001, 0.01)
s = calc_s(t)

s0 = 2.

tt_step = 1
ss_step = 0.05
tt, ss = np.meshgrid(np.arange(t.min()+0.5, t.max()+1, tt_step), np.arange(-.5*s0, 1.5*s0+0.001, ss_step))
dsdt_tt = np.ones(tt.shape)
dsdt_ss = calc_dsdt(tt, ss)

smin, smax = -0.1, 1.4

plt.figure()
plt.plot(t[750]+plot_shift, s[750], 'ro', linewidth=1.)
plt.plot(t+plot_shift, s, 'k:', linewidth=1.)
#plt.plot(t[::t_end]+10, s[::t_end], 'ro', linewidth=1.)
plt.quiver(tt+plot_shift, ss, dsdt_tt, dsdt_ss, scale=0.06,
        angles='xy', color='#999999', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=1)
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.xlim(-0.5, 8.5)
plt.ylim(smin, smax)
plt.savefig('fd_plot.pdf')
plt.close()

plt.figure()
plt.plot(t[750]+plot_shift, s[750], 'ro', linewidth=1.)
plt.plot(t+plot_shift, s, 'k:', linewidth=1.)
#plt.plot(t[::t_end]+10, s[::t_end], 'ro', linewidth=1.)
plt.quiver(tt+plot_shift, ss, dsdt_tt, dsdt_ss, scale=0.08,
        angles='xy', color='#999999', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=1)
plt.plot(x_expl_1+plot_shift, y_expl_1, 'k+--')
plt.plot(x_expl_2+plot_shift, y_expl_2, 'k+--')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.xlim(-0.5, 8.5)
plt.ylim(smin, smax)
plt.savefig('fd_answer_1.pdf')

plt.figure()
plt.plot(t[750]+plot_shift, s[750], 'ro', linewidth=1.)
plt.plot(t+plot_shift, s, 'k:', linewidth=1.)
#plt.plot(t[::t_end]+10, s[::t_end], 'ro', linewidth=1.)
plt.quiver(tt+plot_shift, ss, dsdt_tt, dsdt_ss, scale=0.08,
        angles='xy', color='#999999', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=1)
plt.plot(x_impl_1+plot_shift, y_impl_1, 'k+--')
plt.plot(x_impl_2+plot_shift, y_impl_2, 'k+--')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.xlim(-0.5, 8.5)
plt.ylim(smin, smax)
plt.savefig('fd_answer_2.pdf')

plt.figure()
plt.plot(t[750]+plot_shift, s[750], 'ro', linewidth=1.)
plt.plot(t+plot_shift, s, 'k:', linewidth=1.)
#plt.plot(t[::t_end]+10, s[::t_end], 'ro', linewidth=1.)
plt.quiver(tt+plot_shift, ss, dsdt_tt, dsdt_ss, scale=0.08,
        angles='xy', color='#999999', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=1)
plt.plot(x_mixed_1+plot_shift, y_mixed_1, 'k+--')
plt.plot(x_mixed_2+plot_shift, y_mixed_2, 'k+--')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.xlim(-0.5, 8.5)
plt.ylim(smin, smax)
plt.savefig('fd_answer_3.pdf')
plt.close()
