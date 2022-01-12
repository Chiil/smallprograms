abstract type Boundary end

struct BoundaryDefault <: Boundary
    a::Array{Float64, 2}
    b::Array{Float64, 2}

    BoundaryDefault(itot, jtot) = new(zeros(itot, jtot), zeros(itot, jtot))
end

struct BoundarySurface <: Boundary
    a::Array{Float64, 2}
    b::Array{Float64, 2}
    c::Array{Float64, 2}

    BoundarySurface(itot, jtot) = new(zeros(itot, jtot), zeros(itot, jtot), zeros(itot, jtot))
end

function process_surface(b::BoundaryDefault)
end

function process_surface(b::BoundarySurface)
    b.c[:, :] .+= 100
end

function process_boundaries(b::Boundary)
    b.a[:, :] .+= 1
    b.b[:, :] .+= 10

    process_surface(b)
end

b_default = BoundaryDefault(2, 2)
b_surface = BoundarySurface(2, 2)

process_boundaries(b_default)
process_boundaries(b_surface)

println("b_default: $b_default")
println("b_surface: $b_surface")

