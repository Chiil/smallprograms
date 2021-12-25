## This is the module that will be part of the package.
module Kernels

export @make_kernel

macro make_kernel(do_a, do_b, do_c)
    ex_rhs_list = []

    if do_a
        push!(ex_rhs_list, :(log.(a[:])))
    end

    if do_b
        push!(ex_rhs_list, :(- sin.(b[:])))
    end

    if do_c
        push!(ex_rhs_list, :(- cos.(c[:])))
    end

    if length(ex_rhs_list) == 0
        ex = quote 
            function kernel!(at, a, b, c)
            end
        end
    else
        if length(ex_rhs_list) == 1
            ex_rhs = ex_rhs_list[1]
        else
            ex_rhs = Expr(:call, :+, ex_rhs_list...)
        end

        ex = quote 
            function kernel!(at, a, b, c)
                @. at[:] += $ex_rhs
            end
        end
    end

    print(ex)
    return esc(ex)
end

end


## This is the script on the user side.
using BenchmarkTools
using .Kernels

n = 2^16
at = rand(n); a = rand(n); b = rand(n); c = rand(n)

do_a = true; do_b = true; do_c = true

@eval @make_kernel($do_a, $do_b, $do_c)

@btime kernel!($at, $a, $b, $c)
