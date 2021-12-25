## This is the module that will be part of the package.
module Kernels

export kernel!

macro make_kernel()
    ex = quote 
        function kernel!(at, a, b, c)
            at[:] += log.(a[:]) + sin.(b[:]) + cos.(c[:])
        end
    end

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
