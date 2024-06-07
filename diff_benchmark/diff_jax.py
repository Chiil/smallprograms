import jax.numpy as jnp
from jax import jit

from timeit import default_timer as timer

itot = 384;
jtot = 384;
ktot = 384;

@jit
def diff(at, a, visc, dxidxi, dyidyi, dzidzi):
    at_new = at.at[1:ktot-1, 1:jtot-1, 1:itot-1].add(
            visc * (
                + ( (a[1:ktot-1, 1:jtot-1, 2:itot  ] - a[1:ktot-1, 1:jtot-1, 1:itot-1])
                  - (a[1:ktot-1, 1:jtot-1, 1:itot-1] - a[1:ktot-1, 1:jtot-1, 0:itot-2]) ) * dxidxi
                + ( (a[1:ktot-1, 2:jtot  , 1:itot-1] - a[1:ktot-1, 1:jtot-1, 1:itot-1])
                  - (a[1:ktot-1, 1:jtot-1, 1:itot-1] - a[1:ktot-1, 0:jtot-2, 1:itot-1]) ) * dyidyi
                + ( (a[2:ktot  , 1:jtot-1, 1:itot-1] - a[1:ktot-1, 1:jtot-1, 1:itot-1])
                  - (a[1:ktot-1, 1:jtot-1, 1:itot-1] - a[0:ktot-2, 1:jtot-1, 1:itot-1]) ) * dzidzi
                )
            )

    return at_new

float_type = jnp.float32
# float_type = jnp.float64

nloop = 10;
ncells = itot*jtot*ktot;

at = jnp.zeros((ktot, jtot, itot), dtype=float_type)

index = jnp.arange(ncells, dtype=float_type)
a = (index/(index+1))**2
del(index)
a = a.reshape(ktot, jtot, itot)

# Check results
c = float_type(0.1)
at = diff(at, a, c, c, c, c).block_until_ready()
print("at={0}".format(at.flatten()[itot*jtot+itot+itot//2]))

# Time the loop
start = timer()
for i in range(nloop):
    at = diff(at, a, c, c, c, c).block_until_ready()
end = timer()

print("Time/iter: {0} s ({1} iters)".format((end-start)/nloop, nloop))
