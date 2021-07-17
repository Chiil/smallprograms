using BenchmarkTools

function diff(
        at::Array{Float64, 3}, a::Array{Float64, 3},
        visc::Float64, dxidxi::Float64, dyidyi::Float64, dzidzi::Float64,
        itot::Int64, jtot::Int64, ktot::Int64)

    @threads for k in 2:ktot-1
        for j in 2:jtot-1
            @simd for i in 2:itot-1
                @inbounds at[i, j, k] += visc * (
                    (a[i-1, j, k] - 2. * a[i, j, k] + a[i+1, j, k]) * dxidxi +
                    (a[i, j-1, k] - 2. * a[i, j, k] + a[i, j+1, k]) * dyidyi +
                    (a[i, j, k-1] - 2. * a[i, j, k] + a[i, j, k+1]) * dzidzi )
            end
        end
    end
end

itot = 384
jtot = 384
ktot = 384

a = Array{Float64, 3}(undef, itot, jtot, ktot)
at = Array{Float64, 3}(undef, itot, jtot, ktot)

visc = 0.1
dxidxi = 0.1
dyidyi = 0.1
dzidzi = 0.1

@btime diff(
        at, a,
        visc, dxidxi, dyidyi, dzidzi,
        itot, jtot, ktot)

@btime diff(
        at, a,
        visc, dxidxi, dyidyi, dzidzi,
        itot, jtot, ktot)
