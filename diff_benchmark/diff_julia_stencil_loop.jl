## Packages
using BenchmarkTools
using LoopVectorization

## Macros
function make_index(a, arrays, i, j, k)
    if a in arrays
        if i < 0
            i_int = convert(Int, abs(i))
            ex_i = :( i-$i_int )
        elseif i > 0 
            i_int = convert(Int, abs(i))
            ex_i = :( i+$i_int )
        else
            ex_i = :( i )
        end

        if j < 0
            j_int = convert(Int, abs(j))
            ex_j = :( j-$j_int )
        elseif j > 0 
            j_int = convert(Int, abs(j))
            ex_j = :( j+$j_int )
        else
            ex_j = :( j )
        end

        if k < 0
            k_int = convert(Int, abs(k))
            ex_k = :( k-$k_int )
        elseif k > 0 
            k_int = convert(Int, abs(k))
            ex_k = :( k+$k_int )
        else
            ex_k = :( k )
        end

        return :( $a[ $ex_i, $ex_j, $ex_k] )
    else
        return :( $a )
    end
end

function process_expr(ex, arrays, i, j, k)
    n = 1

    if (isa(ex.args[1], Symbol) && ex.args[1] == Symbol("gradx"))
        ex.args[1] = Symbol("gradx_")
        ex = :( $ex * dxi )
    elseif (isa(ex.args[1], Symbol) && ex.args[1] == Symbol("grady"))
        ex.args[1] = Symbol("grady_")
        ex = :( $ex * dyi )
    elseif (isa(ex.args[1], Symbol) && ex.args[1] == Symbol("gradz"))
        ex.args[1] = Symbol("gradz_")
        ex = :( $ex * dzi )
    end

    args = ex.args
    while n <= length(args)
        if isa(args[n], Expr)
            args[n] = process_expr(args[n], arrays, i, j, k)
            n += 1
        elseif isa(args[n], Symbol)
            if args[n] == Symbol("gradx_")
                if isa(args[n+1], Expr)
                    args[n] = copy(args[n+1])
                    args[n  ] = process_expr(args[n  ], arrays, i+0.5, j, k)
                    args[n+1] = process_expr(args[n+1], arrays, i-0.5, j, k)
                elseif isa(args[n+1], Symbol)
                    args[n] = args[n+1]
                    args[n  ] = make_index(args[n  ], arrays, i+0.5, j, k)
                    args[n+1] = make_index(args[n+1], arrays, i-0.5, j, k)
                end
                args[n  ] = :( $(args[n  ])  )
                args[n+1] = :( $(args[n+1])  )
                insert!(args, n, Symbol("-"))
                n += 3
            elseif args[n] == Symbol("grady_")
                if isa(args[n+1], Expr)
                    args[n] = copy(args[n+1])
                    args[n  ] = process_expr(args[n  ], arrays, i, j+0.5, k)
                    args[n+1] = process_expr(args[n+1], arrays, i, j-0.5, k)
                elseif isa(args[n+1], Symbol)
                    args[n] = args[n+1]
                    args[n  ] = make_index(args[n  ], arrays, i, j+0.5, k)
                    args[n+1] = make_index(args[n+1], arrays, i, j-0.5, k)
                end
                args[n  ] = :( $(args[n  ])  )
                args[n+1] = :( $(args[n+1])  )
                insert!(args, n, Symbol("-"))
                n += 3
            elseif args[n] == Symbol("gradz_")
                if isa(args[n+1], Expr)
                    args[n] = copy(args[n+1])
                    args[n  ] = process_expr(args[n  ], arrays, i, j, k+0.5)
                    args[n+1] = process_expr(args[n+1], arrays, i, j, k-0.5)
                elseif isa(args[n+1], Symbol)
                    args[n] = args[n+1]
                    args[n  ] = make_index(args[n  ], arrays, i, j, k+0.5)
                    args[n+1] = make_index(args[n+1], arrays, i, j, k-0.5)
                end
                args[n  ] = :( $(args[n  ])  )
                args[n+1] = :( $(args[n+1])  )
                insert!(args, n, Symbol("-"))
                n += 3
            elseif args[n] == Symbol("interpx")
                if isa(args[n+1], Expr)
                    args[n] = copy(args[n+1])
                    args[n  ] = process_expr(args[n  ], arrays, i+0.5, j, k)
                    args[n+1] = process_expr(args[n+1], arrays, i-0.5, j, k)
                elseif isa(args[n+1], Symbol)
                    args[n] = args[n+1]
                    args[n  ] = make_index(args[n  ], arrays, i+0.5, j, k)
                    args[n+1] = make_index(args[n+1], arrays, i-0.5, j, k)
                end
                args[n  ] = :( 0.5f0 * $(args[n  ]) )
                args[n+1] = :( 0.5f0 * $(args[n+1]) )
                insert!(args, n, Symbol("+"))
                n += 3
 
            else
                args[n] = make_index(args[n], arrays, i, j, k)
                n += 1
            end
        else
            n += 1
        end
    end

    return ex
end

macro fd(arrays, ex)
    ex = process_expr(ex, arrays.args, 0, 0, 0)
    return ex
end

macro fd_loop(ranges, arrays, ex)
    is = ranges.args[1].args[2]; ie = ranges.args[1].args[3]
    js = ranges.args[2].args[2]; je = ranges.args[2].args[3]
    ks = ranges.args[3].args[2]; ke = ranges.args[3].args[3]

    ex = process_expr(ex, arrays.args, 0, 0, 0)

    ex_loop = quote
        @tturbo unroll=8 for k in $ks:$ke
            for j in $js:$je
                for i in $is:$ie
                    $ex
                end
            end
        end
    end
    return ex_loop
end

## Diffusion kernel
function diff!(
        at, a,
        visc, dxi, dyi, dzi,
        itot, jtot, ktot)

    @fd_loop (2:itot-1, 2:jtot-2, 2:ktot-1) (at, a) at += visc * (gradx(gradx(a)) + grady(grady(a)) + gradz(gradz(a)))
end

## Set the grid size.
itot = 384
jtot = 384
ktot = 384

## Solve the problem in double precision.
visc = 0.1
dxi = sqrt(0.1)
dyi = sqrt(0.1)
dzi = sqrt(0.1)

a = rand(Float64, (itot, jtot, ktot))
at = zeros(Float64, (itot, jtot, ktot))

@btime diff!(
        at, a,
        visc, dxi, dyi, dzi,
        itot, jtot, ktot)
