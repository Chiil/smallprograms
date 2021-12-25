## This is the module that will be part of the package.
module Kernels

export kernel!

const do_a = true
const do_b = true
const do_c = true

macro make_kernel()
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

@make_kernel

end


## This is the script on the user side.
using BenchmarkTools
using .Kernels

n = 2^16
at = rand(n); a = rand(n); b = rand(n); c = rand(n)

@btime kernel!($at, $a, $b, $c)
