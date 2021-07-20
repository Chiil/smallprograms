## Macros
macro gradx(arg, offset)
    if offset == "+" || offset == "0"
        e = quote
            ($arg[i+1, j, k] - $arg[i, j, k]) * dxi
        end
        esc(e)
    elseif offset == "-"
        e = quote
            ($arg[i, j, k] - $arg[i-1, j, k]) * dxi
        end
        esc(e)
    end
end

## Packages
using BenchmarkTools
using LoopVectorization

## Diffusion kernel
function diff!(
        at, a,
        visc, dxi, dyi, dzi,
        itot, jtot, ktot)

     @tturbo unroll=8 for k in 2:ktot-1
        for j in 2:jtot-1
            for i in 2:itot-1
                at[i, j, k] += visc * ( (@gradx(a, "+") - @gradx(a, "-") ) )
            end
        end
    end
end

## Set the grid size.
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

@time diff!(
        at, a,
        visc, dxidxi, dyidyi, dzidzi,
        itot, jtot, ktot)

## Solve the problem in single precision.
visc_f = Float32(visc)
dxidxi_f = Float32(dxidxi)
dyidyi_f = Float32(dyidyi)
dzidzi_f = Float32(dzidzi)

a_f = rand(Float32, (itot, jtot, ktot))
at_f = zeros(Float32, (itot, jtot, ktot))

@time diff!(
        at_f, a_f,
        visc_f, dxidxi_f, dyidyi_f, dzidzi_f,
        itot, jtot, ktot)
