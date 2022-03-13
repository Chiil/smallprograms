## Packages.
using Zygote
using Flux
using Flux.Optimise: update!
using PyPlot
pygui(true)


## Constants for learning.
const n_unroll = 4
const n_epoch = 1000
const n_timestep = 10
const learning_rate = 0.01


## Constants for physics.
const visc_ref = 0.4
const dt = 0.25
const dx = 0.5
const dxidxi = 1/(dx^2)
const dxi2 = 1/(2dx)


## Init grid.
x = dx/2:dx:100-dx/2 |> collect


## Init fields.
sigma = 5.
u_ref = @. exp( -(x-20)^2 / sigma^2)
u = copy(u_ref)
u_ref0 = copy(u_ref)


## Functions.
function integrate_ref(u, visc)
    u_gc = [ u[end]; u; u[1] ]
    ul = @view u_gc[1:end-2]; uc = @view u_gc[2:end-1]; ur = @view u_gc[3:end]
    advec = - uc .* (ur .- ul) .* dxi2
    diff = visc .* (ul .- 2uc .+ ur) .* dxidxi
    return uc .+ dt .* (advec .+ diff)
end

m = Chain(
    Conv((7,), 1=>1, leakyrelu),
    Conv((7,), 1=>1, leakyrelu),
    Conv((7,), 1=>1),
)
const m_gc = 3 + 3 + 3

function integrate(u)
    u_ml = [ u[end-m_gc+1:end]; u; u[1:m_gc] ]
    u_ml = reshape(u_ml, (length(u_ml), 1, 1))
    dudt = m(Float32.(u_ml))
    u_new = u .+ dt .* dudt
    return u_new
end


## Loss function.
function loss(u, u_ref)
    loss_sum = 0.
    u_next = copy(u); u_ref_next = copy(u_ref)

    for i in 1:n_unroll
        u_next = integrate(u_next)
        u_ref_next = integrate_ref(u_ref_next, visc_ref)
        loss_sum += sum( (u_next .- u_ref_next).^2 )
    end

    return loss_sum
end


## Set up optimizer.
opt = ADAM(learning_rate)
θ = params(m)


## Optimize steps while integrating model.
for _ in 1:n_timestep
    # Find the optimal velocity and viscosity.
    for i in 1:n_epoch
        grads = gradient(() -> loss(u, u_ref), θ)
        update!(opt, θ, grads)
    end

    # Print and plot status.
    println("params = $θ, loss = $(loss(u, u_ref))")

    # Integrate in time n_unroll steps further to avoid overlap of data.
    # CvH: not sure if this is necessary...
    for i in 1:n_unroll
        u_ref[:] .= integrate_ref(u_ref, visc_ref)
    end

    # Set the field to solve back to the reference field.
    u[:] .= u_ref[:]
end


## Run freely.
for _ in 1:n_timestep
    u[:] .= integrate(u)
    u_ref[:] .= integrate_ref(u_ref, visc_ref)
end


## Plot the actual values.
plt.figure()
plt.subplot(211)
plt.plot(x, u, "C0-")
plt.plot(x, u_ref, "k:")
plt.plot(x, u_ref0, linestyle=":", color="#bbbbbb")
plt.subplot(212)
plt.plot(x, u - u_ref, "C1-")
