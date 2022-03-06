## Packages.
using Zygote
using Flux
using Flux.Optimise: update!
using PyPlot
pygui(true)


## Init arrays
x = 0:1.0:100 |> collect
sigma = 5.
y_ref = @. exp( -(x-50)^2 / sigma^2)
y = copy(y_ref)
y_ref0 = copy(y_ref)


## Function to optimize.
const visc_ref = 0.5
const dt = 1
const dx = x[2] - x[1]
const dxi2 = 1/(dx^2)

function integrate(y, visc)
    yl = @view y[1:end-2]; yc = @view y[2:end-1]; yr = @view y[3:end]
    y_new = yc .+ dt .* visc .* (yl .- 2yc .+ yr) .* dxi2
    return vcat(y_new[1], y_new, y_new[end])
end


## Loss function.
function loss(y, y_ref)
    return sum( (integrate(y, visc_param[1]) .- integrate(y_ref, visc_ref)).^2 )
end


## Set up optimizer.
visc_param = [0.001]
opt = ADAM(0.1)
θ = params(visc_param)


## Optimize steps.
y_ref[:] .= integrate(y_ref, visc_ref)
y[:] .= integrate(y, visc_param[1])

for i in 1:100
    grads = gradient(() -> loss(y, y_ref), θ)
    for p in (visc_param,)
        update!(opt, p, grads[p])
    end
end

println("visc_param = $visc_param, loss = $(loss(y, y_ref))")

plt.figure()
plt.subplot(211)
plt.plot(x, y, "C0-")
plt.plot(x, y_ref, "k:")
plt.plot(x, y_ref0, color="#bbbbbb")
plt.subplot(211)
plt.plot(x, y - y_ref, "C1-")
