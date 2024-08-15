import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial

from timeit import default_timer as timer


# Numpy-like operation.

@jit
def interp2(a, b):
    return 0.5*(a + b);

@partial(jit, static_argnums=(8, 9, 10))
def diff(at, a, u, v, w, dxi, dyi, dzi, itot, jtot, ktot):
    i_c = jnp.s_[1:ktot-1, 1:jtot-1, 1:itot-1]
    i_w = jnp.s_[1:ktot-1, 1:jtot-1, 0:itot-2]
    i_e = jnp.s_[1:ktot-1, 1:jtot-1, 2:itot  ]
    i_s = jnp.s_[1:ktot-1, 0:jtot-2, 1:itot-1]
    i_n = jnp.s_[1:ktot-1, 2:jtot  , 1:itot-1]
    i_b = jnp.s_[0:ktot-2, 1:jtot-1, 1:itot-1]
    i_t = jnp.s_[2:ktot  , 1:jtot-1, 1:itot-1]

    at_new = at.at[i_c].add(
            - ( u[i_e]*interp2(a[i_c], a[i_e])
              - u[i_c]*interp2(a[i_w], a[i_c]) ) * dxi
            - ( v[i_n]*interp2(a[i_c], a[i_n])
              - v[i_c]*interp2(a[i_s], a[i_c]) ) * dyi
            - ( w[i_t]*interp2(a[i_c], a[i_t])
              - w[i_c]*interp2(a[i_b], a[i_c]) ) * dzi
            )

    return at_new


itot = 770;
jtot = 770;
ktot = 770;

float_type = jnp.float32

nloop = 30;
ncells = itot*jtot*ktot;

dxi = float_type(0.1)
dyi = float_type(0.1)
dzi = float_type(0.1)

@jit
def init_a(index):
    return (index/(index+1))**2

at = jnp.zeros((ktot, jtot, itot), dtype=float_type)
index = jnp.arange(ncells, dtype=float_type)
a = init_a(index)
u = init_a(index)
v = init_a(index)
w = init_a(index)
del(index)

a = a.reshape(ktot, jtot, itot)
u = u.reshape(ktot, jtot, itot)
v = v.reshape(ktot, jtot, itot)
w = w.reshape(ktot, jtot, itot)

at = diff(at, a, u, v, w, dxi, dyi, dzi, itot, jtot, ktot).block_until_ready()
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//2]))

# Time the loop
start = timer()
for i in range(nloop):
    at = diff(at, a, u, v, w, dxi, dyi, dzi, itot, jtot, ktot).block_until_ready()
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//4]))
