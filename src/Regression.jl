# https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/

module Regression


import MLJModelInterface
const MMI = MLJModelInterface
import MLJBase: matrix, fit, predict, fitted_params
using Distances: pairwise, SqEuclidean


export RBFRegression, IDWRegression, fit, partial_fit, predict, fitted_params


const δ = 1e-6  # small number to avoid nans
const sqeuclideandist = SqEuclidean()

@inline _tomat(X) = begin
    ndim = ndims(X)
    if ndim == 0
        X = [X;;]
    elseif ndim == 1
        X = reshape(X, :, 1)
    end
    return matrix(X)
end


_kernelfunctions::Dict{Symbol,Function} = Dict(
    :inversequadratic => (d², ε) -> 1 / (1 + ε^2 * d²),
    :multiquadric => (d², ε) -> √(1 + ε^2 * d²),
    :linear => (d², ε) -> ε * √d²,
    :gaussian => (d², ε) -> exp(-ε^2 * d²),
    :thinplatespline => (d², ε) -> ε^2 * d² * log(ε * √d² .+ δ),
    :inversemultiquadric => (d², ε) -> 1 / √(1 + ε^2 * d²),
)


"""
Regression with radial basis function (RBF) kernels.

# Fields
- kernel: type of RBF function to use. One of `:inversequadratic`, `:multiquadric`,
  `:linear`, `:gaussian`, `:thinplatespline`, `:inversemultiquadric`.
- ϵ: parameter to scale distances.
"""
MMI.@mlj_model mutable struct RBFRegression <: MMI.Deterministic
    kernel::Symbol = :inversequadratic::(_ in (
        :inversequadratic,
        :multiquadric,
        :linear,
        :gaussian,
        :thinplatespline,
        :inversemultiquadric,
    ))
    ϵ::Float64 = 1e-3::(_ >= 0)
end


function fit(m::RBFRegression, X, y)
    # process inputs
    #  - X ∈ (n_samples, n_features)
    #  - y ∈ (n_samples, 1)
    Xm = _tomat(X)
    ym = _tomat(y)

    # create matrix of kernel evaluations
    d² = pairwise(SqEuclidean(), Xm, dims=1)
    M = _kernelfunctions[m.kernel].(d², m.ϵ)

    # compute coefficients
    coefs = M \ ym
    return (coefs, Xm), nothing, NamedTuple{}()
end


fitted_params(m::RBFRegression, (coefs, _)) = (coefs=coefs,)


function predict(m::RBFRegression, (coefs, Xm), Xnew)
    # process input
    Xnewm = _tomat(Xnew)

    # create matrix of kernel evaluations
    d² = pairwise(sqeuclideandist, Xm, Xnewm, dims=1)
    M = _kernelfunctions[m.kernel].(d², m.ϵ)

    # predict as linear combination
    ynew = M' * coefs
    return ynew
end


# TODO: partial_fit for RBFRegression


"""
Regression with inverse distance weighting (IDW).

# Fields
- weighting: type of weighting function to use. One of `:inversesquared`,
  `:expinversesquared`.
"""
MMI.@mlj_model mutable struct IDWRegression <: MMI.Deterministic
    weighting::Symbol = :inversesquared::(_ in (:inversesquared, :expinversesquared))
end


function fit(m::IDWRegression, X, y)
    # process inputs
    #  - X ∈ (n_samples, n_features)
    #  - y ∈ (n_samples, 1)
    Xm = _tomat(X)
    ym = _tomat(y)

    # do nothing
    return (Xm, ym), nothing, NamedTuple{}()
end


fitted_params(m::IDWRegression, fitresults) = ()


function predict(m::IDWRegression, (Xm, ym), Xnew)
    # process input
    Xnewm = _tomat(Xnew)

    # create matrix of weights
    d = pairwise(sqeuclideandist, Xm, Xnewm, dims=1)
    W = 1 ./ (d .+ δ)
    if m.weighting == :expinversesquared
        W .*= exp.(-d)
    end

    # predict as weighted average
    ynew = (W' * ym) ./ sum(W, dims=1)'
    return ynew
end


function partial_fit(m::IDWRegression, fitresults, X, y)
    if fitresults === nothing
        return fit(m, X, y)
    end
    Xm, ym = fitresults

    # process inputs
    #  - X ∈ (n_samples, n_features)
    #  - y ∈ (n_samples, 1)
    Xp = _tomat(X)  # p for partial
    yp = _tomat(y)

    # append new data to old matrices and return
    return ([Xm; Xp], [ym; yp]), nothing, NamedTuple{}()
end

end
