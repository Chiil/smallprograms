## Packages.
using Flux
using Flux.Optimise: update!
using PyPlot
pygui(true)


## Init array.
a = 0.5; n = 4.
x = 0.5:1:99.5 |> collect
y_ref = a*sin.(n * 2pi/100 * x) .+ 0.1 .* (rand(length(x)) .- 0.5)


## Function to optimize.
a_param = rand(1)
n_param = 10*rand(1)

function sin_fit(x)
    y = a_param[1] * sin.(n_param[1] * 2pi/100 * x)
end

## Plot initial data
plt.figure()
plt.plot(x, y_ref, "C3o")
# loss(x, y_ref) = sum( (sin_fit(x) .- y_ref).^2 )
# loss(x, y_ref) = sum( abs.(sin_fit(x) .- y_ref) )
function loss(x, y_ref)
    idxs = rand(1:length(x), 10)
    sum( abs.(sin_fit(x[idxs]) .- y_ref[idxs]) )
end


## Set up optimizer.
println(loss(x, y_ref))
plt.plot(x, sin_fit.(x), "k:")


## Set up optimizer.
opt = ADAM(0.05)
θ = params(a_param, n_param)

for i in 1:1000
    grads = gradient(() -> loss(x, y_ref), θ)
    for p in (a_param, n_param)
        update!(opt, p, grads[p])
    end
end
println("a_param = $a_param, n_param = $n_param")
println(loss(x, y_ref))
plt.plot(x, sin_fit.(x))
