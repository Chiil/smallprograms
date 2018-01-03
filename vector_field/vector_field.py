import matplotlib.pyplot as plt
import numpy as np

s0 = 100.
a = -0.006
p = 0.01

def calc_s(t):
    return (s0+p/a)*np.exp(a*t) - p/a

def calc_dsdt(t, s):
    return a*s + p*np.sin(2.*np.pi/365*t)

def calc_s_impl(t_start, t_end, s_start):
    dt = t_end - t_start
    return -(s_start + dt*p) / (a*dt - 1.)

t = np.arange(0, 1001, 1.)
s = calc_s(t)

tt_step = 100
ss_step = 0.25*s0
tt, ss = np.meshgrid(np.arange(0, t.max()+1, tt_step), np.arange(-.5*s0, 1.5*s0+0.001, ss_step))
dsdt_tt = np.ones(tt.shape)
dsdt_ss = calc_dsdt(tt, ss)

def trendline(t_start, t_end, s_start, at_start=True):
    dt = t_end - t_start
    if at_start:
        s_end = s_start + calc_dsdt(t_start, s_start)*dt
    else:
        s_end = calc_s_impl(t_start, t_end, s_start)

    return np.array([t_start, t_end]), np.array([s_start, s_end])

t_end = 250
x_expl_1, y_expl_1 = trendline(0, t_end, calc_s(0))
x_expl_2, y_expl_2 = trendline(t_end, 2*t_end, y_expl_1[-1])
x_expl_3, y_expl_3 = trendline(2*t_end, 3*t_end, y_expl_2[-1])
x_expl_4, y_expl_4 = trendline(3*t_end, 4*t_end, y_expl_3[-1])

x_impl_1, y_impl_1 = trendline(0, t_end, calc_s(0), False)
x_impl_2, y_impl_2 = trendline(t_end, 2*t_end, y_impl_1[-1], False)
x_impl_3, y_impl_3 = trendline(2*t_end, 3*t_end, y_impl_2[-1], False)
x_impl_4, y_impl_4 = trendline(3*t_end, 4*t_end, y_impl_3[-1], False)

smin, smax = -0.6*s0, 1.6*s0

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('vectorfield_1.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('vectorfield_2.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.plot(x_expl_1, y_expl_1, 'k+--')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('vectorfield_3.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.plot(x_expl_1, y_expl_1, 'k+--')
plt.plot(x_expl_2, y_expl_2, 'k+--')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('vectorfield_4.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.plot(x_expl_1, y_expl_1, 'k+--')
plt.plot(x_expl_2, y_expl_2, 'k+--')
plt.plot(x_impl_1, y_impl_1, 'k+:')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('vectorfield_5.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.plot(x_expl_1, y_expl_1, 'k+--')
plt.plot(x_expl_2, y_expl_2, 'k+--')
plt.plot(x_impl_1, y_impl_1, 'k+:')
plt.plot(x_impl_2, y_impl_2, 'k+:')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('vectorfield_6.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.plot(x_expl_1, y_expl_1, 'k+--')
plt.plot(x_expl_2, y_expl_2, 'k+--')
plt.plot(x_expl_3, y_expl_3, 'k+--')
plt.plot(x_expl_4, y_expl_4, 'k+--')
plt.plot(x_impl_1, y_impl_1, 'k+:')
plt.plot(x_impl_2, y_impl_2, 'k+:')
plt.plot(x_impl_3, y_impl_3, 'k+:')
plt.plot(x_impl_4, y_impl_4, 'k+:')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('vectorfield_7.pdf')
plt.close()

