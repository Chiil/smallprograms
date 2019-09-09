import numpy as np
import matplotlib.pyplot as plt

itot_in = 8
jtot_in = 8
index_in = 3600

i_multi = 16
j_multi = 16
index_out = 0

ktot = 144

field_list = ["u"] #, "v", "w", "thl", "qt", "qr", "qs", "qg"]

# CvH: hack a field to test
u = np.random.rand(ktot, jtot_in, itot_in)
u.tofile("{0:}.{1:07d}".format("u", index_in))
# CvH: end hack

def aggregate_xy(a_in):
    a_out = np.tile(a_in, (1, j_multi, i_multi))
    return a_out

# Loop over the list.
for field in field_list:
    a_in = np.fromfile("{0:}.{1:07d}".format(field, index_in))
    a_in.shape = (ktot, jtot_in, itot_in)

    a_out = aggregate_xy(a_in)
    a_out.tofile("{0:}.{1:07d}".format(field, index_out))

# Plot it.
u = np.fromfile("u.0000000")
u.shape = (ktot, jtot_in * j_multi, itot_in * i_multi)

k_plot = ktot // 2

plt.figure()
plt.pcolormesh(u[k_plot])
plt.colorbar()
plt.show()
