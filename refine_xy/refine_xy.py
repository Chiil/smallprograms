import numpy as np
import matplotlib.pyplot as plt

itot_in = 96
itot_out = 480

jtot_in = 96
jtot_out = 480

ktot = 144

field_list = []
field_list.append(("u", True, False))
field_list.append(("v", False, True))
field_list.append(("w", False, False))
field_list.append(("thl", False, False))
field_list.append(("qt", False, False))
field_list.append(("qr", False, False))
field_list.append(("qs", False, False))
field_list.append(("qg", False, False))

# The input range contains ghost cells cover full range.
dx_in = 1. / itot_in
xh_in = np.arange(0., 1.+dx_in/2, dx_in)
x_in = np.arange(-dx_in/2, 1.+dx_in, dx_in)

dy_in = 1. / jtot_in
yh_in = np.arange(0., 1.+dy_in/2, dy_in)
y_in = np.arange(-dy_in/2, 1.+dy_in, dy_in)

# The output range does not contain ghost cells.
dx_out = 1. / itot_out
xh_out = np.arange(0., 1.-dx_out/2, dx_out)
x_out = np.arange(dx_out/2, 1., dx_out)

dy_out = 1. / jtot_out
yh_out = np.arange(0., 1.-dy_out/2, dy_out)
y_out = np.arange(dy_out/2, 1., dy_out)

# Create transformation matrix.
def make_matrix(s_in, s_out):
    M = np.zeros((s_out.size, s_in.size))
    
    # Find indices surrounding point.
    for i in range(itot_out):
        i_end = np.argmax(s_in > s_out[i])
        i_start = i_end-1
    
        fac_start = (s_in[i_end] - s_out[i]) / (s_in[i_end] - s_in[i_start])
        fac_end = 1. - fac_start
    
        M[i, i_start] = fac_start
        M[i, i_end] = fac_end

    return M

M_x  = make_matrix( x_in,  x_out)
M_xh = make_matrix(xh_in, xh_out)
M_y  = make_matrix( y_in,  y_out)
M_yh = make_matrix(yh_in, yh_out)

def refine_xy(a_in, at_xh, at_yh):
    i_start = 0 if at_xh else 1
    j_start = 0 if at_yh else 1

    a_in_gc = np.empty((ktot, jtot_in+1+j_start, itot_in+1+i_start))
    a_in_gc[:, j_start:-1, i_start:-1] = a_in[:, :, :]

    a_out = np.einsum('ijk,lk->ijl', a_in_gc, M_xh, optimize=True) if at_xh else np.einsum('ijk,lk->ijl', a_in_gc, M_x, optimize=True)
    a_out = np.einsum('ijk,lj->ilk', a_out  , M_yh, optimize=True) if at_yh else np.einsum('ijk,lj->ilk', a_out  , M_y, optimize=True)

    return a_out

# Loop over the list.
for name, at_xh, at_yh in field_list:
    a_in = np.fromfile("{0:}.{1:07d}".format(name, 0))
    a_in.shape = (ktot, jtot_in, itot_in)

    a_out = refine_xy(a_in, at_xh, at_yh)
    a_out.tofile("{0:}.{:07d}.new".format(name, 0))

"""
# Some arrays to test it.
u_in = np.empty((ktot, jtot_in, itot_in))
s_in = np.empty((ktot, jtot_in, itot_in))
u_in[:,:,:] = np.sin(2.*np.pi*xh_in[None, None,  :-1]) * np.sin(2.*np.pi*y_in[None, 1:-1, None])
s_in[:,:,:] = np.sin(2.*np.pi* x_in[None, None, 1:-1]) * np.sin(2.*np.pi*y_in[None, 1:-1, None])

u_out = refine_xy(u_in, True, False)
s_out = refine_xy(u_in, False, False)

# Plot it.
k_plot = ktot // 2

plt.figure()
plt.subplot(121)
plt.pcolormesh(xh_in[:-1], y_in[1:-1], u_in[k_plot])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(xh_out, y_out, u_out[k_plot])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.colorbar()

plt.show()
"""
