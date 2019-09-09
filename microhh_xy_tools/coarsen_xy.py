import numpy as np
import matplotlib.pyplot as plt

itot_in = 16
jtot_in = 16
index_in = 3600

itot_out = 7
jtot_out = 7
index_out = 0

ktot = 144

field_list = ["u"] #, "v", "w", "thl", "qt", "qr", "qs", "qg"]

dx = 1. / itot_in
dy = 1. / jtot_in

xh = np.arange(0, 1., dx)
yh = np.arange(0, 1., dy)

x = np.arange(dx/2, 1., dx)
y = np.arange(dy/2, 1., dy)

# Define the coarse grid.
dxc = 1. / itot_out
dyc = 1. / jtot_out

xhc = np.arange(0, 1., dxc)
yhc = np.arange(0, 1., dyc)

xc = np.arange(dxc/2, 1., dxc)
yc = np.arange(dyc/2, 1., dyc)

# CvH: hack a field to test
u0 = np.empty((ktot, jtot_in, itot_in))
u0[:,:,:] = np.sin(2.*np.pi*xh[None, None, :]) * np.sin(2.*np.pi*y[None, :, None]) + np.arange(ktot)[:, None, None]
u0.tofile("{0:}.{1:07d}".format("u", index_in))
# CvH: end hack

## COARSENING PROCEDURE STARTS HERE.
def make_trans_matrix(x_edge, xc_edge, merge_outer_control_volumes):
    # Create the mapping matrix from fine to coarse.
    map_x = np.zeros((xc_edge.size-1, x_edge.size-1))
    
    # Check for each cell to which extent it contributes and add the weight to mapping matrix.
    map_x[:, :] = np.maximum(0, np.minimum(xc_edge[1:, None], x_edge[None, 1:]) - np.maximum(xc_edge[:-1, None], x_edge[None, :-1]))

    dist_x = (xc_edge[1:] - xc_edge[:-1])
    
    # Create a mapping matrix for period data that is one element shorter in both axes.
    periodic_x = True
    if merge_outer_control_volumes:
        map_xp = map_x[:-1, :-1]
        map_xp[:, 0] += map_x[:-1, -1]
        map_xp[0, :] += map_x[-1, :-1]
        map_xp[0, 0] += map_x[-1, -1]

        dist_xp = dist_x[:-1]
        dist_xp[0] += dist_x[-1]

        return map_xp / dist_xp[:, None]

    else:
        return map_x / dist_x[:, None]

# Define the edges
xc_edge = np.arange(0, 1., dxc)
xc_edge = np.append(xc_edge, 1.)

x_edge = np.arange(0, 1., dx)
x_edge = np.append(x_edge, 1.)

xhc_edge = np.arange(dxc/2, 1., dxc)
xhc_edge = np.append(0, xhc_edge)
xhc_edge = np.append(xhc_edge, 1.)

xh_edge = np.arange(dx/2, 1., dx)
xh_edge = np.append(0, xh_edge)
xh_edge = np.append(xh_edge, 1.)

map_x  = make_trans_matrix(x_edge , xc_edge , False)
map_xh = make_trans_matrix(xh_edge, xhc_edge, True )

yc_edge = np.arange(0, 1., dyc)
yc_edge = np.append(yc_edge, 1.)

y_edge = np.arange(0, 1., dy)
y_edge = np.append(y_edge, 1.)

yhc_edge = np.arange(dyc/2, 1., dyc)
yhc_edge = np.append(0, yhc_edge)
yhc_edge = np.append(yhc_edge, 1.)

yh_edge = np.arange(dy/2, 1., dy)
yh_edge = np.append(0, yh_edge)
yh_edge = np.append(yh_edge, 1.)

map_y  = make_trans_matrix(y_edge , yc_edge , False)
map_yh = make_trans_matrix(yh_edge, yhc_edge, True )

# Loop over the list.
for field in field_list:
    a_in = np.fromfile("{0:}.{1:07d}".format(field, index_in))
    a_in.shape = (ktot, jtot_in, itot_in)

    a_out = np.einsum('ijk,lk->ijl', a_in , map_xh if field == "u" else map_x, optimize=True)
    a_out = np.einsum('ijk,lj->ilk', a_out, map_yh if field == "v" else map_y, optimize=True)

    a_out.tofile("{0:}.{1:07d}".format(field, index_out))

# Plot it.
u = np.fromfile("u.0000000")
u.shape = (ktot, jtot_out, itot_out)

k_plot = ktot // 2

plt.figure()
plt.subplot(221)
plt.pcolormesh(xh, y, u0[k_plot])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.colorbar()

plt.subplot(222)
plt.pcolormesh(xhc, yc, u[k_plot])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.colorbar()

plt.subplot(223)
plt.plot(xh , u0[k_plot, 2, :])
plt.plot(xhc, u [k_plot, 2, :])

plt.subplot(224)
plt.plot(u0.mean(axis=(1,2)))
plt.plot(u .mean(axis=(1,2)))

plt.show()
