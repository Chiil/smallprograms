## Packages.
using Zygote
using Flux
using Flux.Optimise: update!
using PyPlot
pygui(true)


## Init arrays.
x = 0:0.5:100 |> collect
sigma = 5.
y_ref = @. exp( -(x-20)^2 / sigma^2)
y = copy(y_ref)
y_ref0 = copy(y_ref)


## Constants for learning.
const n_unroll = 4
const n_epoch = 1000
const n_timestep = 10
const learning_rate = 0.1


## Constants for physics.
const u_ref = 0.8
const visc_ref = 0.4
const dt = 0.25
const dx = x[2] - x[1]
const dxidxi = 1/(dx^2)
const dxi2 = 1/(2dx)
println("CFL = $(u_ref*dt/dx), dn = $(visc_ref*dt/dx^2)")


## Functions.
function integrate_ref(y, u, visc)
    yl = @view y[1:end-2]; yc = @view y[2:end-1]; yr = @view y[3:end]
    advec = - u .* (yr .- yl) .* dxi2
    diff = visc .* (yl .- 2yc .+ yr) .* dxidxi
    y_new = yc .+ dt .* (advec .+ diff)

    # Return statement includes a neumann BC of zero on both sides.
    return vcat(y_new[1], y_new, y_new[end])
end

mlmodel = Chain(
    Conv((3,), 1=>1, identity)
)

function integrate(y)
    y_ml = reshape(y, (length(y), 1, 1))
    dydt = mlmodel(Float32.(y_ml))
    yc = @view y[2:end-1]
    y_new = yc .+ dt .* dydt
    return vcat(y_new[1], y_new, y_new[end])
end


## Loss function.
function loss(y, y_ref)
    loss_sum = 0.
    y_next = copy(y); y_ref_next = copy(y_ref)

    for i in 1:n_unroll
        y_next = integrate(y_next)
        y_ref_next = integrate_ref(y_ref_next, u_ref, visc_ref)
        loss_sum += sum( (y_next .- y_ref_next).^2 )
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
        grads = gradient(() -> loss(y, y_ref), θ)
        update!(opt, θ, grads)
    end

    # Print and plot status.
    println("params = $θ, loss = $(loss(y, y_ref))")

    # Integrate in time.
    y[:] .= integrate(y)
    y_ref[:] .= integrate_ref(y_ref, u_ref, visc_ref)
end


## Plot the actual values.
plt.figure()
plt.subplot(211)
plt.plot(x, y, "C0-")
plt.plot(x, y_ref, "k:")
plt.plot(x, y_ref0, linestyle=":", color="#bbbbbb")
plt.subplot(212)
plt.plot(x, y - y_ref, "C1-")