## Packages.
using Zygote
using Flux
using Flux.Optimise: update!
using PyPlot
pygui(true)


## Init arrays.
x = 0:0.5:100 |> collect
sigma = 5.
u_ref = @. exp( -(x-20)^2 / sigma^2)
u = copy(u_ref)
u_ref0 = copy(u_ref)


## Constants for learning.
const n_unroll = 4
const n_epoch = 1000
const n_timestep = 10
const learning_rate = 0.1


## Constants for physics.
const visc_ref = 0.4
const dt = 0.25
const dx = x[2] - x[1]
const dxidxi = 1/(dx^2)
const dxi2 = 1/(2dx)
println("CFL = $(u_ref*dt/dx), dn = $(visc_ref*dt/dx^2)")


## Functions.
function integrate_ref(u, visc)
    ul = @view u[1:end-2]; uc = @view u[2:end-1]; ur = @view u[3:end]
    advec = - uc .* (ur .- ul) .* dxi2
    diff = visc .* (ul .- 2uc .+ ur) .* dxidxi
    u_new = uc .+ dt .* (advec .+ diff)

    # Return statement includes a neumann BC of zero on both sides.
    return vcat(u_new[1], u_new, u_new[end])
end

mlmodel = Chain(
    Conv((3,), 1=>1, identity)
)

function integrate(u)
    u_ml = reshape(u, (length(u), 1, 1))
    dudt = mlmodel(Float32.(u_ml))
    uc = @view u[2:end-1]
    u_new = uc .+ dt .* dudt
    return vcat(u_new[1], u_new, u_new[end])
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
θ = params(mlmodel)


## Optimize steps while integrating model.
for _ in 1:n_timestep
    # Find the optimal velocity and viscosity.
    for i in 1:n_epoch
        grads = gradient(() -> loss(u, u_ref), θ)
        update!(opt, θ, grads)
    end

    # Print and plot status.
    println("params = $θ, loss = $(loss(u, u_ref))")

    # Integrate in time.
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