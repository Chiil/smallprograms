## Macros
function make_notation(a, arrays, i, j, k)
    if a in arrays
        return :( $a[i+$i, j+$j, k+$k] )
    else
        return :( $a )
    end
end

function process_rhs(args, arrays, i, j, k)
    for n in size(args)
        arg = args[n]
        if typeof(arg) == Expr
            args[n].args = process_rhs(arg.args, arrays, i, j, k)
        elseif typeof(arg) == Symbol
            #if symbol == "gradx"
                args[n] = make_notation(arg, arrays, i, j, k)
            # else
            #     args[n] = make_notation(arg, arrays, i, j, k)
            # end
        else
            throw(ArgumentError(arg, "Dunnowhatodo"))
        end
    end
    return args[:]
end

macro fd(arrays, e)
    i = 0; j = 0; k = 0;

    eo = copy(e)

    # Set the left hand side
    eo.args[1] = make_notation(eo.args[1], arrays.args, i, j, k)

    # Recursively work through args
    eo.args[2:end] = process_rhs(eo.args[2:end], arrays.args, i, j, k)

    dump(e)
    dump(eo)

    eo_loop = quote
        @tturbo unroll=8 for k in 2:ktot-1
            for j in 2:jtot-1
                for i in 2:itot-1
                    $eo
                end
            end
        end
    end
    println(eo_loop)
end

## Packages
using BenchmarkTools
using LoopVectorization

## Diffusion kernel
function diff!(
        at::AbstractArray{T, 3}, a::AbstractArray{T, 3},
        visc, dxi, dyi, dzi,
        itot, jtot, ktot) where T <: Number

    @fd (at, a) at += gradx( a )
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
