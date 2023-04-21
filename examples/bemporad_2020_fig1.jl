# reproduces Figure 1 of Bemporad (2020)

include("../src/NonMyopicGlobOpt.jl")
include("../src/TestCases.jl")
using .NonMyopicGlobOpt: RBFRegression, IDWRegression, partial_fit, predict
import .TestCases
using Plots


# grab function
f = TestCases.bemporad2020

# create initial data points
X = [-0.883, 0.667]
Y = f.(X)

# create regressors and fit to initial data
regressors = [
    IDWRegression(),
    RBFRegression(ϵ=0.5),
    RBFRegression(kernel=:thinplatespline, ϵ=0.01),
]
fitresults = [partial_fit(r, nothing, X, Y)[1] for r in regressors]

# add new datapoints to the regressions
for x in [-0.633, -0.2, 0.127]
    local y = f(x)
    global X = vcat(X, x)
    global Y = vcat(Y, y)
    for i in eachindex(regressors)
        fitresults[i] = partial_fit(regressors[i], fitresults[i], x, y)[1]
    end
end

# now that we fitted 5 datapoints, use the regressors to predict over all the domain
n = 100
x = range(-1, +1, n)
z = f.(x)
plt = plot(x, z, xlims=(-1, +1), ylims=(0, 2.5), ls=:dash, label="f(x)", legend=:top)
plot!(plt, X, Y, seriestype=:scatter, label=nothing)
for (r, fr) in zip(regressors, fitresults)
    local z_hat = predict(r, fr, x)
    local lbl = r isa IDWRegression ? "IDW" : "RBF $(r.kernel), ϵ=$(r.ϵ)"
    plot!(plt, x, z_hat, label=lbl)
end
display(plt)
