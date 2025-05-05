import matplotlib.pyplot as plt
import numpy as np

alpha = 1.0
beta = 1.0

c0 = (alpha**3 + 4 * alpha - 2 * beta) / (alpha**3 + 4 * alpha)


def calc_s(t):
    return np.exp(-alpha * t) * (
        c0
        - beta
        * (
            2 * alpha * np.exp(alpha * t) * np.sin(2 * t)
            + alpha**2 * np.exp(alpha * t) * np.cos(2 * t)
            + (-(alpha**2) - 4) * np.exp(alpha * t)
        )
        / (2 * alpha**3 + 8 * alpha)
    )


def calc_s_impl(t_start, t_end, s_start):
    dt = t_end - t_start
    return (s_start + dt * beta * np.sin(t_end) ** 2) / (1.0 + dt * alpha)


def calc_dsdt(t, s):
    return -alpha * s + beta * np.sin(t) ** 2


def trendline(t_start, t_end, s_start, method):
    dt = t_end - t_start
    if method == "expl":
        s_end = s_start + calc_dsdt(t_start, s_start) * dt
    elif method == "impl":
        s_end = calc_s_impl(t_start, t_end, s_start)
    elif method == "mixed_impl":
        s_mid = s_start + calc_dsdt(t_start, s_start) * dt / 2.0
        s_end = calc_s_impl(t_start + dt / 2, t_end, s_mid)

    return np.array([t_start, t_end]), np.array([s_start, s_end])


t_step = 0.5 * np.arange(5)
x_expl_1, y_expl_1 = trendline(t_step[0], t_step[1], calc_s(t_step[0]), "expl")
x_expl_2, y_expl_2 = trendline(t_step[1], t_step[2], y_expl_1[-1], "expl")
x_expl_3, y_expl_3 = trendline(t_step[2], t_step[3], y_expl_2[-1], "expl")
x_expl_4, y_expl_4 = trendline(t_step[3], t_step[4], y_expl_3[-1], "expl")

x_impl_1, y_impl_1 = trendline(t_step[0], t_step[1], calc_s(t_step[0]), "impl")
x_impl_2, y_impl_2 = trendline(t_step[1], t_step[2], y_impl_1[-1], "impl")
x_impl_3, y_impl_3 = trendline(t_step[2], t_step[3], y_impl_2[-1], "impl")
x_impl_4, y_impl_4 = trendline(t_step[3], t_step[4], y_impl_3[-1], "impl")

x_mixed_1, y_mixed_1 = trendline(t_step[0], t_step[1], calc_s(t_step[0]), "mixed_impl")
x_mixed_2, y_mixed_2 = trendline(t_step[1], t_step[2], y_mixed_1[-1], "mixed_impl")
x_mixed_3, y_mixed_3 = trendline(t_step[2], t_step[3], y_mixed_2[-1], "mixed_impl")
x_mixed_4, y_mixed_4 = trendline(t_step[3], t_step[4], y_mixed_3[-1], "mixed_impl")

t = np.arange(0.0, 3.001, 0.01)
s = calc_s(t)

s0 = 2.0

tt_step = 0.25
ss_step = 0.05
tt, ss = np.meshgrid(
    np.arange(t.min(), t.max() + 1, tt_step),
    np.arange(-0.5 * s0, 1.5 * s0 + 0.001, ss_step),
)
dsdt_tt = np.ones(tt.shape)
dsdt_ss = calc_dsdt(tt, ss)

smin, smax = -0.1, 1.1

plt.figure(figsize=(6, 3.5))
plt.plot(t[0], s[0], "ro", linewidth=1.0)
plt.plot(t, s, "k:", linewidth=1.0)
plt.quiver(
    tt,
    ss,
    dsdt_tt,
    dsdt_ss,
    scale=0.06,
    angles="xy",
    color="#999999",
    pivot="mid",
    units="dots",
    headlength=0,
    headaxislength=0,
    width=1,
)
plt.xlabel("t (d)")
plt.ylabel("h (m)")
plt.xlim(-0.15, 2.15)
plt.ylim(smin, smax)
plt.tight_layout()
plt.savefig("fd_plot.pdf")

plt.figure(figsize=(6, 3.5))
plt.plot(t[0], s[0], "ro", linewidth=1.0)
plt.plot(t, s, "k:", linewidth=1.0)
plt.quiver(
    tt,
    ss,
    dsdt_tt,
    dsdt_ss,
    scale=0.06,
    angles="xy",
    color="#999999",
    pivot="mid",
    units="dots",
    headlength=0,
    headaxislength=0,
    width=1,
)
plt.plot(x_expl_1, y_expl_1, "k+--")
plt.plot(x_expl_2, y_expl_2, "k+--")
plt.plot(x_expl_3, y_expl_3, "k+--")
plt.plot(x_expl_4, y_expl_4, "k+--")
plt.xlabel("t (d)")
plt.ylabel("h (m)")
plt.xlim(-0.15, 2.15)
plt.ylim(smin, smax)
plt.tight_layout()
plt.savefig("fd_answer_1.pdf")

plt.figure(figsize=(6, 3.5))
plt.plot(t[0], s[0], "ro", linewidth=1.0)
plt.plot(t, s, "k:", linewidth=1.0)
plt.quiver(
    tt,
    ss,
    dsdt_tt,
    dsdt_ss,
    scale=0.06,
    angles="xy",
    color="#999999",
    pivot="mid",
    units="dots",
    headlength=0,
    headaxislength=0,
    width=1,
)
plt.plot(x_impl_1, y_impl_1, "k+--")
plt.plot(x_impl_2, y_impl_2, "k+--")
plt.plot(x_impl_3, y_impl_3, "k+--")
plt.plot(x_impl_4, y_impl_4, "k+--")
plt.xlabel("t (d)")
plt.ylabel("h (m)")
plt.xlim(-0.15, 2.15)
plt.ylim(smin, smax)
plt.tight_layout()
plt.savefig("fd_answer_2.pdf")

plt.figure(figsize=(6, 3.5))
plt.plot(t[0], s[0], "ro", linewidth=1.0)
plt.plot(t, s, "k:", linewidth=1.0)
plt.quiver(
    tt,
    ss,
    dsdt_tt,
    dsdt_ss,
    scale=0.06,
    angles="xy",
    color="#999999",
    pivot="mid",
    units="dots",
    headlength=0,
    headaxislength=0,
    width=1,
)
plt.plot(x_mixed_1, y_mixed_1, "k+--")
plt.plot(x_mixed_2, y_mixed_2, "k+--")
plt.plot(x_mixed_3, y_mixed_3, "k+--")
plt.plot(x_mixed_4, y_mixed_4, "k+--")
plt.xlabel("t (d)")
plt.ylabel("h (m)")
plt.xlim(-0.15, 2.15)
plt.ylim(smin, smax)
plt.tight_layout()
plt.savefig("fd_answer_3.pdf")
