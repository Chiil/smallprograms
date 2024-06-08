import jax
import jax.numpy as jnp
from jax import jit

from timeit import default_timer as timer

itot = 512;
jtot = 512;
ktot = 512;

i_c = jnp.s_[1:ktot-1, 1:jtot-1, 1:itot-1]
i_w = jnp.s_[1:ktot-1, 1:jtot-1, 0:itot-2]
i_e = jnp.s_[1:ktot-1, 1:jtot-1, 2:itot  ]
i_s = jnp.s_[1:ktot-1, 0:itot-2, 1:jtot-1]
i_n = jnp.s_[1:ktot-1, 2:itot  , 1:jtot-1]
i_b = jnp.s_[0:itot-2, 1:ktot-1, 1:jtot-1]
i_t = jnp.s_[2:itot  , 1:ktot-1, 1:jtot-1]


@jit
def diff(at, a, visc, dxidxi, dyidyi, dzidzi):
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


float_type = jnp.float32

nloop = 30;
ncells = itot*jtot*ktot;

at = jnp.zeros((ktot, jtot, itot), dtype=float_type)

index = jnp.arange(ncells, dtype=float_type)

@jit
def init_a(index):
    return (index/(index+1))**2

a = init_a(index)
del(index)

a = a.reshape(ktot, jtot, itot)

# Check results
c = float_type(0.1)

a_gpu = jax.device_put(a)
at_gpu = jax.device_put(at)

at_gpu = diff(at_gpu, a_gpu, c, c, c, c).block_until_ready()

at = jax.device_get(at_gpu)
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//2]))

# Time the loop
start = timer()
for i in range(nloop):
    at_gpu = diff(at_gpu, a_gpu, c, c, c, c).block_until_ready()
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))

at = jax.device_get(at_gpu)
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//4]))

