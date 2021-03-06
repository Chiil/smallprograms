# !/usr/bin/python
import numpy as np


# Input parameters.
theta0 = 300.
u0 = 5.
z0m = 0.1
z0h = 0.1
zsl = 5.

Q_net = 500.


# Constants.
kappa = 0.4
g = 9.81
theta_ref = 300.
rho = 1.2
cp = 1005.


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


def solve_theta_s(theta_s):
    while True:
        # Evaluate the surface energy balance error.
        L, ustar = solve_sl(theta_s)
        theta_star = fhw(L) * (theta0 - theta_s)
        H = - rho * cp * ustar * theta_star
        L_out = 5.67e-8 * theta_s**4
        eval_seb = Q_net - H - L_out
        ra = 1./(fmw(L)*fhw(L)*u0)

        print('theta_s = {:4f}, z/L = {:4f}, ustar = {:4f}, ra = {:4f}, H = {:4f}, L_out = {:4f}, error = {:4f}'.format(theta_s, zsl/L, ustar, ra, H, L_out, eval_seb))

        if abs(eval_seb) < 1.e-8:
            return theta_s

        # Estimate the slope.
        eps = 1.e-8
        theta_s_eps = theta_s + eps
        L_eps, ustar_eps = solve_sl(theta_s_eps)
        theta_star_eps = fhw(L_eps) * (theta0 - theta_s_eps)
        H_eps = - rho * cp * ustar_eps * theta_star_eps
        L_out = 5.67e-8 * theta_s_eps**4
        eval_seb_eps = Q_net - H_eps - L_out
        slope = (eval_seb_eps - eval_seb) / eps

        dtheta = - eval_seb / slope

        max_step = 10.
        theta_s += max(-max_step, min(dtheta, max_step))


# Solve for the surface temperature, use the atmospheric temperature as the starting point.
solve_theta_s(theta0)
