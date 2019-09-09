import numpy as np
import matplotlib.pyplot as plt

itot_in = 8
jtot_in = 8
index_in = 3600

itot_out = 16
jtot_out = 16
index_out = 0

ktot = 144

field_list = ["u"] #, "v", "w", "thl", "qt", "qr", "qs", "qg"]

# CvH: hack a field to test
u = np.ones((ktot, jtot_in, itot_in))
u.tofile("{0:}.{1:07d}".format("u", index_in))
# CvH: end hack

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

    # Set the ghost cells.
    if at_xh:
        a_in_gc[:, :, -1] = a_in_gc[:, :, 0]
    else:
        a_in_gc[:, :,  0] = a_in_gc[:, :, -2]
        a_in_gc[:, :, -1] = a_in_gc[:, :,  1]
 
    if at_yh:
        a_in_gc[:, -1, :] = a_in_gc[:, 0, :]
    else:
        a_in_gc[:,  0, :] = a_in_gc[:, -2, :]
        a_in_gc[:, -1, :] = a_in_gc[:,  1, :]

    a_out = np.einsum('ijk,lk->ijl', a_in_gc, M_xh, optimize=True) if at_xh else np.einsum('ijk,lk->ijl', a_in_gc, M_x, optimize=True)
    a_out = np.einsum('ijk,lj->ilk', a_out  , M_yh, optimize=True) if at_yh else np.einsum('ijk,lj->ilk', a_out  , M_y, optimize=True)

    return a_out

# Loop over the list.
for field in field_list:
    a_in = np.fromfile("{0:}.{1:07d}".format(field, index_in))
    a_in.shape = (ktot, jtot_in, itot_in)

    a_out = refine_xy(a_in, field == "u", field == "v")
    a_out.tofile("{0:}.{1:07d}".format(field, index_out))

# Plot it.
u = np.fromfile("u.0000000")
u.shape = (ktot, jtot_out, itot_out)

k_plot = ktot // 2

plt.figure()
plt.pcolormesh(xh_out, y_out, u[k_plot])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.colorbar()

plt.show()
