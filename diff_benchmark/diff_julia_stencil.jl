## Macros
function make_notation(a, arrays, i, j, k)
    if a in arrays
        return :( $a[i+$i, j+$j, k+$k] )
    else
        return :( $a )
    end
end

function process_rhs(args, arrays, i, j, k)
    dump(args)
    for n in size(args)
        if typeof(args[n]) == Expr
            args[n] = process_rhs(args[n].args, arrays, i, j, k)
        elseif typeof(args[n]) == Symbol
            args[n] = make_notation(args[n], arrays, i, j, k)
            println("Symbol: ", args[n])
        else
            throw(ArgumentError(args[n], "Dunnowhatodo"))
        end
    end
    dump(args)
    return args
end

macro fd(arrays, e)
    dump(e)
    i = 0; j = 0; k = 0;

    # Set the left hand side
    lhs = make_notation(e.args[1], arrays.args, i, j, k)

    # Store the head
    op = e.head

    # Construct the RHS from the tree
    rhs_args = copy(e.args[2:end])

    # Recursively work through args
    rhs = process_rhs(rhs_args, arrays.args, i, j, k)

    eo_stencil = Expr(op, lhs, rhs)
    eo = quote
        @tturbo unroll=8 for k in 2:ktot-1
            for j in 2:jtot-1
                for i in 2:itot-1
                    $eo_stencil
                end
            end
        end
    end
    print(eo)
end

## Packages
using BenchmarkTools
using LoopVectorization

## Diffusion kernel
function diff!(
        at::AbstractArray{T, 3}, a::AbstractArray{T, 3},
        visc, dxi, dyi, dzi,
        itot, jtot, ktot) where T <: Number

    @fd (at, a) at += visc * a
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
