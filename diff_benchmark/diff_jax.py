import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from timeit import default_timer as timer


# Numpy-like operation.
@partial(jit, static_argnums=(6, 7, 8))
def diff(at, a, visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot):
    i_c = jnp.s_[1:ktot-1, 1:jtot-1, 1:itot-1]
    i_w = jnp.s_[1:ktot-1, 1:jtot-1, 0:itot-2]
    i_e = jnp.s_[1:ktot-1, 1:jtot-1, 2:itot  ]
    i_s = jnp.s_[1:ktot-1, 0:jtot-2, 1:itot-1]
    i_n = jnp.s_[1:ktot-1, 2:jtot  , 1:itot-1]
    i_b = jnp.s_[0:ktot-2, 1:jtot-1, 1:itot-1]
    i_t = jnp.s_[2:ktot  , 1:jtot-1, 1:itot-1]

    at_new = at.at[i_c].add(
            visc * (
                + ( (a[i_e] - a[i_c])
                  - (a[i_c] - a[i_w]) ) * dxidxi
                + ( (a[i_n] - a[i_c])
                  - (a[i_c] - a[i_s]) ) * dyidyi
                + ( (a[i_t] - a[i_c])
                  - (a[i_c] - a[i_b]) ) * dzidzi
                )
            )

    return at_new


# Using convolve.
@partial(jit, static_argnums=(3, 4, 5))
def diff2(at, a, diff_weights, itot, jtot, ktot):

    at_diff = jax.scipy.signal.convolve(a, diff_weights, mode='valid')
    at_new = at.at[1:ktot-1, 1:jtot-1, 1:itot-1].add(at_diff)

    return at_new


itot = 64;
jtot = 64;
ktot = 64;

float_type = jnp.float32

nloop = 30;
ncells = itot*jtot*ktot;

dxidxi = float_type(0.1)
dyidyi = float_type(0.1)
dzidzi = float_type(0.1)
visc = float_type(0.1)

@jit
def init_a(index):
    return (index/(index+1))**2


## FIRST EXPERIMENT.
at = jnp.zeros((ktot, jtot, itot), dtype=float_type)
index = jnp.arange(ncells, dtype=float_type)
a = init_a(index)
del(index)
a = a.reshape(ktot, jtot, itot)

at = diff(at, a, visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot).block_until_ready()
print("(first check) at={0}".format(at.flatten()[itot*jtot+itot+itot//2]))

# Time the loop
start = timer()
for i in range(nloop):
    at = diff(at, a, visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot).block_until_ready()
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//4]))


## SECOND EXPERIMENT.
at = jnp.zeros((ktot, jtot, itot), dtype=float_type)
index = jnp.arange(ncells, dtype=float_type)
a = init_a(index)
del(index)
a = a.reshape(ktot, jtot, itot)

diff_weights = jnp.array(
        [
            [
                [ 0, 0, 0],
                [ 0, dzidzi, 0],
                [ 0, 0, 0]
            ],
            [
                [ 0, dyidyi, 0],
                [ dxidxi, -(dxidxi + dyidyi + dzidzi), dxidxi],
                [ 0, dyidyi, 0]
            ],
            [
                [ 0, 0, 0],
                [ 0, dzidzi, 0],
                [ 0, 0, 0]
            ]
        ])

diff_weights *= visc

print(diff_weights[0, :, :])

at = diff2(at, a, diff_weights, itot, jtot, ktot).block_until_ready()
print("(first check) at={0}".format(at.flatten()[itot*jtot+itot+itot//2]))

# Time the loop
start = timer()
for i in range(nloop):
    at = diff2(at, a, diff_weights, itot, jtot, ktot).block_until_ready()
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//4]))

