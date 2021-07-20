using BenchmarkTools
using LoopVectorization


function diff_view!(
        at::AbstractArray{T}, a::AbstractArray{T},
        visc, dxidxi, dyidyi, dzidzi,
        itot, jtot, ktot) where T <: Number

    at_c = view(at, 2:itot-1, 2:jtot-1, 2:ktot-1)

    a_c = view(a, 2:itot-1, 2:jtot-1, 2:ktot-1)
    a_w = view(a, 1:itot-2, 2:jtot-1, 2:ktot-1)
    a_e = view(a, 3:itot  , 2:jtot-1, 2:ktot-1)
    a_s = view(a, 2:itot-1, 1:jtot-2, 2:ktot-1)
    a_n = view(a, 2:itot-1, 3:jtot  , 2:ktot-1)
    a_b = view(a, 2:itot-1, 2:jtot-1, 1:ktot-2)
    a_t = view(a, 2:itot-1, 2:jtot-1, 3:ktot  )

    @tturbo unroll=8 @. at_c += visc * (
            (a_w - T(2) * a_c + a_e) * dxidxi +
            (a_s - T(2) * a_c + a_n) * dyidyi +
            (a_b - T(2) * a_c + a_t) * dzidzi )
end


itot = 384
jtot = 384
ktot = 384


# Solve the problem in double precision.
a = rand(Float64, (itot, jtot, ktot))
at = zeros(Float64, (itot, jtot, ktot))

visc = 0.1
dxidxi = 0.1
dyidyi = 0.1
dzidzi = 0.1


# Solve the problem in single precision.
@btime diff_view!(
        $at, $a,
        $visc, $dxidxi, $dyidyi, $dzidzi,
        $itot, $jtot, $ktot)

a_f = rand(Float32, (itot, jtot, ktot))
at_f = zeros(Float32, (itot, jtot, ktot))

visc_f = Float32(visc)
dxidxi_f = Float32(dxidxi)
dyidyi_f = Float32(dyidyi)
dzidzi_f = Float32(dzidzi)

@btime diff_view!(
        $at_f, $a_f,
        $visc_f, $dxidxi_f, $dyidyi_f, $dzidzi_f,
        $itot, $jtot, $ktot)
