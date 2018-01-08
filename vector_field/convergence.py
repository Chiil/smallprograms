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

tt_step = 100
ss_step = 0.25*s0
tt, ss = np.meshgrid(np.arange(0, t.max()+1, tt_step), np.arange(-.5*s0, 1.5*s0+0.001, ss_step))
dsdt_tt = np.ones(tt.shape)
dsdt_ss = calc_dsdt(tt, ss)

def calc_expl_series(dt):
    t = np.arange(0, 1001, dt)
    s = np.zeros(t.shape)
    s[0] = s0
    for i in range(1, t.shape[0]):
        s[i] = (1 + dt*a)*s[i-1]
    return t, s

def calc_impl_series(dt):
    t = np.arange(0, 1001, dt)
    s = np.zeros(t.shape)
    s[0] = s0
    for i in range(1, t.shape[0]):
        s[i] = 1./(1 - dt*a)*s[i-1]
    return t, s

def calc_midpoint_series(dt):
    t = np.arange(0, 1001, dt)
    s = np.zeros(t.shape)
    s[0] = s0
    for i in range(1, t.shape[0]):
        s[i] = (1.+dt*a/2)/(1.-dt*a/2)*s[i-1]
    return t, s

t_e_200, s_e_200 = calc_expl_series(200)
t_e_100, s_e_100 = calc_expl_series(100)
t_e_050, s_e_050 = calc_expl_series(50)
t_e_025, s_e_025 = calc_expl_series(25)
expl_errors_200 = sum( (s_e_200 - calc_s(t_e_200))**2 )**.5 / t_e_200.size
expl_errors_100 = sum( (s_e_100 - calc_s(t_e_100))**2 )**.5 / t_e_100.size
expl_errors_050 = sum( (s_e_050 - calc_s(t_e_050))**2 )**.5 / t_e_050.size
print(expl_errors_200, expl_errors_100, expl_errors_050)

t_i_200, s_i_200 = calc_impl_series(200)
t_i_100, s_i_100 = calc_impl_series(100)
t_i_050, s_i_050 = calc_impl_series(50)
impl_errors_200 = sum( (s_i_200 - calc_s(t_i_200))**2 )**.5 / t_i_200.size
impl_errors_100 = sum( (s_i_100 - calc_s(t_i_100))**2 )**.5 / t_i_100.size
impl_errors_050 = sum( (s_i_050 - calc_s(t_i_050))**2 )**.5 / t_i_050.size
print(impl_errors_200, impl_errors_100, impl_errors_050)

t_m_200, s_m_200 = calc_midpoint_series(200)
t_m_100, s_m_100 = calc_midpoint_series(100)
t_m_050, s_m_050 = calc_midpoint_series(50)
m_errors_200 = sum( abs(s_m_200 - calc_s(t_m_200))**1 )**1 / t_m_200.size
m_errors_100 = sum( abs(s_m_100 - calc_s(t_m_100))**1 )**1 / t_m_100.size
m_errors_050 = sum( abs(s_m_050 - calc_s(t_m_050))**1 )**1 / t_m_050.size
print(m_errors_200, m_errors_100, m_errors_050)

smin, smax = -0.6*s0, 1.1*s0
t_end = 200

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
plt.plot(t[::t_end], s[::t_end], 'ro', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('convergence_1.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
plt.plot(t[::t_end], s[::t_end], 'ro', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.plot(t_e_200, s_e_200, 'k+:')
plt.plot(t_e_100, s_e_100, 'k+--')
plt.plot(t_e_050, s_e_050, 'k+-')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('convergence_2.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
plt.plot(t[::t_end], s[::t_end], 'ro', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.plot(t_i_200, s_i_200, 'k+:')
plt.plot(t_i_100, s_i_100, 'k+--')
plt.plot(t_i_050, s_i_050, 'k+-')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('convergence_3.pdf')
plt.close()

plt.figure()
plt.plot(t, s, 'r-', linewidth=1.5)
plt.plot(t[::t_end], s[::t_end], 'ro', linewidth=1.5)
#plt.streamplot(tt, ss, dsdt_tt, dsdt_ss,
#        color='#bbbbbb')
plt.quiver(tt, ss, dsdt_tt, dsdt_ss,
        angles='xy', color='#dddddd', pivot='mid', units='dots',
        headlength=0, headaxislength=0, width=2)
plt.plot(t_m_200, s_m_200, 'k+:')
plt.plot(t_m_100, s_m_100, 'k+--')
plt.plot(t_m_050, s_m_050, 'k+-')
plt.xlabel('t (d)')
plt.ylabel('h (m)')
plt.ylim(smin, smax)
plt.savefig('convergence_4.pdf')
plt.close()

