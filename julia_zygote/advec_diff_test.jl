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


## Constants.
const visc_ref = 0.4
const u_ref = 0.8
const dt = 0.25
const dx = x[2] - x[1]
const dxidxi = 1/(dx^2)
const dxi2 = 1/(2dx)
println("CFL = $(u_ref*dt/dx), dn = $(visc_ref*dt/dx^2)")


## Function to optimize.
function integrate(y, u, visc)
    yl = @view y[1:end-2]; yc = @view y[2:end-1]; yr = @view y[3:end]
    advec = - u .* (yr .- yl) .* dxi2
    diff = visc .* (yl .- 2yc .+ yr) .* dxidxi
    y_new = yc .+ dt .* (advec .+ diff)
    return vcat(y_new[1], y_new, y_new[end])
end


## Loss function.
function loss(y, y_ref)
    loss_sum = 0.

    y_next = integrate(y, u_param[1], visc_param[1])
    y_ref_next = integrate(y_ref, u_ref, visc_ref)
    loss_sum += sum( (y_next .- y_ref_next).^2 )

    for i in 2:10
        y_next = integrate(y_next, u_param[1], visc_param[1])
        y_ref_next = integrate(y_ref_next, u_ref, visc_ref)
        loss_sum += sum( (y_next .- y_ref_next).^2 )
    end

    return loss_sum
end


## Set up optimizer.
u_param = [0.]
visc_param = [0.]
opt = ADAM(0.1)
θ = params(u_param, visc_param)


## Optimize steps.
# Find the optimal velocity and viscosity.
for i in 1:20
    grads = gradient(() -> loss(y, y_ref), θ)
    for p in (u_param, visc_param,)
        update!(opt, p, grads[p])
    end
end

# Print and plot status.
println("u_param = $u_param, visc_param = $visc_param, loss = $(loss(y, y_ref))")

# Integrate in time.
y[:] .= integrate(y, u_param[1], visc_param[1])
y_ref[:] .= integrate(y_ref, u_ref, visc_ref)

plt.figure()
plt.subplot(211)
plt.plot(x, y, "C0-")
plt.plot(x, y_ref, "k:")
plt.plot(x, y_ref0, linestyle=":", color="#bbbbbb")
plt.subplot(212)
plt.plot(x, y - y_ref, "C1-")

