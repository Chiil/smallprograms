function make_index(a, arrays, i)
    if a in arrays
        if i < 0
            i_abs = abs(i)
            return :( $a[i-$i_abs] )
        elseif i > 0 
            return :( $a[i+$i] )
        else
            return :( $a[i] )
        end
    else
        return :( $a )
    end
end


function process_expr_args(args, arrays, i)
    n = 1
    while n <= length(args)
        if isa(args[n], Expr)
            process_expr_args(args[n].args, arrays, i)
            n += 1
        elseif isa(args[n], Symbol)
            if args[n] == Symbol("gradx")
                args[n] = args[n+1]
                if isa(args[n], Expr)
                    process_expr_args(args[n  ].args, arrays, i+1)
                    process_expr_args(args[n+1].args, arrays, i-1)
                elseif isa(args[n], Symbol)
                    args[n  ] = make_index(args[n  ], arrays, i+1)
                    args[n+1] = make_index(args[n+1], arrays, i-1)
                end
                insert!(args, n, Symbol("-"))
                break 
            else
                args[n] = make_index(args[n], arrays, i)
                n += 1
            end
        else
            n += 1
        end
    end
end

macro fd(arrays, ex)
    println(ex)
    process_expr_args(ex.args, arrays.args, 0)
    println(ex)
end

@fd (at, a) at += gradx( gradx( a ) )
