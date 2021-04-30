# !/usr/bin/python
import numpy as np


# Constants.
kappa = 0.4
g = 9.81
theta_ref = 300.
rho = 1.2
cp = 1005.
Lv = 2.5e6
sigma = 5.67e-8


# Input parameters.
theta0 = 290.
q0 = 0.005
u0 = 0.001
z0m = 0.1
z0h = 0.01
zsl = 10.
albedo = 0.2
emis = 0.8
rs = 100.

S_in = 0.
S_out = albedo * S_in
L_in = emis * sigma * theta0**4
p = 1.e5

rs = rs if S_in > 0. else 1e8


# Integrated flux gradient relationships following Wilson.
def psimw(zeta):
    if(zeta <= 0):
        x = (1. + 3.6 * abs(zeta) ** (2. / 3.)) ** (-0.5)
        psimw = 3. * np.log((1. + 1. / x) / 2.)
    else:
        psimw = -2. / 3. * (zeta - 5. / 0.35) * \
            np.exp(-0.35 * zeta) - zeta - (10. / 3.) / 0.35
    return psimw


def psihw(zeta):
    if(zeta <= 0):
        x = (1. + 7.9 * abs(zeta) ** (2. / 3.)) ** (-0.5)
        psihw = 3. * np.log((1. + 1. / x) / 2.)
    else:
        psihw = -2. / 3. * (zeta - 5. / 0.35) * np.exp(-0.35 * zeta) - \
            (1. + (2. / 3.) * zeta) ** (1.5) - (10. / 3.) / 0.35 + 1.
    return psihw


def fmw(L):
    return kappa / (np.log(zsl / z0m) - psimw(zsl / L) + psimw(z0m / L))


def fhw(L):
    return kappa / (np.log(zsl / z0h) - psihw(zsl / L) + psihw(z0h / L))


def eval_w(L):
    return zsl / L * fmw(L)**2 / fhw(L)


zL = np.linspace(-10000., 10., 100000)
L = zsl / zL


# Evaluate the function (this has to be done only once for a range of Ri).
eval0_w = np.zeros(L.size)
for i in range(L.size):
    eval0_w[i] = eval_w(L[i])


def solve_sl(theta_s):
    db0 = g/theta_ref * (theta0 - theta_s)

    # Find the value that matches with the value of Ri.
    Ri = zsl * kappa * db0 / u0**2
    
    if (max(eval0_w - Ri) * min(eval0_w - Ri) > 0):
        zL0_w = np.nan if (Ri < 0) else 10.
    else:
        zL0_w = np.interp(0., eval0_w - Ri, zL)
    B0_w = -zL0_w * fmw(zsl / zL0_w)**3 * u0**3 / (kappa * zsl)
    
    ustar_w = u0 * fmw(zsl / zL0_w)

    # Return the Wilson variant.
    return zsl/zL0_w, ustar_w


def e_sat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))


def q_sat(T):
    return 0.622 * e_sat(T) / p


def solve_theta_s(theta_s):
    while True:
        # Evaluate the surface energy balance error.
        L, ustar = solve_sl(theta_s)
        theta_star = fhw(L) * (theta0 - theta_s)
        ra = 1./(fmw(L)*fhw(L)*u0)

        H = - rho * cp * ustar * theta_star
        LE = rho * Lv / (ra + rs) * (q_sat(theta_s) - q0)
        L_out = sigma * theta_s**4
        eval_seb = S_in - S_out + L_in - L_out - H - LE

        print('theta_s = {:4f}, z/L = {:4f}, ustar = {:4f}, ra = {:4f}, H = {:4f}, LE = {:4f}, L_out = {:4f}, error = {:4f}'.format(theta_s, zsl/L, ustar, ra, H, LE, L_out, eval_seb))

        if abs(eval_seb) < 1.e-8:
            return theta_s

        # Estimate the slope.
        eps = 1.e-8
        theta_s_eps = theta_s + eps
        L_eps, ustar_eps = solve_sl(theta_s_eps)
        theta_star_eps = fhw(L_eps) * (theta0 - theta_s_eps)
        ra_eps = 1./(fmw(L_eps)*fhw(L_eps)*u0)

        H_eps = - rho * cp * ustar_eps * theta_star_eps
        LE_eps = rho * Lv / (ra_eps + rs) * (q_sat(theta_s_eps) - q0)
        L_out_eps = 5.67e-8 * theta_s_eps**4
        eval_seb_eps = S_in - S_out + L_in - L_out_eps - H_eps - LE_eps
        slope = (eval_seb_eps - eval_seb) / eps

        dtheta = - eval_seb / slope

        max_step = 10.
        theta_s += max(-max_step, min(dtheta, max_step))


# Solve for the surface temperature, use the atmospheric temperature as the starting point.
solve_theta_s(theta0)
