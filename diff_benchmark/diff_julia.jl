## Packages
using BenchmarkTools
using LoopVectorization

## Diffusion kernel
function diff!(
        at, a,
        visc, dxidxi, dyidyi, dzidzi,
        itot, jtot, ktot)

    @tturbo unroll=8 for k in 2:ktot-1
        for j in 2:jtot-1
            for i in 2:itot-1
                at[i, j, k] += visc * (
                    (a[i-1, j  , k  ] - 2.f0 * a[i, j, k] + a[i+1, j  , k  ]) * dxidxi +
                    (a[i  , j-1, k  ] - 2.f0 * a[i, j, k] + a[i  , j+1, k  ]) * dyidyi +
                    (a[i  , j  , k-1] - 2.f0 * a[i, j, k] + a[i  , j  , k+1]) * dzidzi )
            end
        end
    end
end

# Set the grid size.
itot = 384
jtot = 384
ktot = 384

## Solve the problem in double precision.
visc = 0.1
dxidxi = 0.1
dyidyi = 0.1
dzidzi = 0.1

a = rand(Float64, (itot, jtot, ktot))
at = zeros(Float64, (itot, jtot, ktot))

@btime diff!(
        $at, $a,
        $visc, $dxidxi, $dyidyi, $dzidzi,
        $itot, $jtot, $ktot)

## Solve the problem in single precision.
visc_f = Float32(visc)
dxidxi_f = Float32(dxidxi)
dyidyi_f = Float32(dyidyi)
dzidzi_f = Float32(dzidzi)

a_f = rand(Float32, (itot, jtot, ktot))
at_f = zeros(Float32, (itot, jtot, ktot))

@btime diff!(
        $at_f, $a_f,
        $visc_f, $dxidxi_f, $dyidyi_f, $dzidzi_f,
        $itot, $jtot, $ktot)
