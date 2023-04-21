# https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/

import MLJModelInterface
const MMI = MLJModelInterface
import MLJBase: matrix, fit, predict, fitted_params
using Distances: pairwise
include("./consts.jl")


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


function fit(m::RBFRegression, X, y; minv::Bool=false)
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

    # if requested, compute also the inverse of M
    Minv = minv ? inv(M) : nothing
    return (coefs, Xm, ym, Minv), nothing, NamedTuple{}()
end


fitted_params(m::RBFRegression, fitresults) = (coefs=fitresults[1],)


function predict(m::RBFRegression, (coefs, Xm, _, _), Xnew)
    # process input
    Xnewm = _tomat(Xnew)

    # create matrix of kernel evaluations
    d² = pairwise(sqeuclideandist, Xm, Xnewm, dims=1)
    M = _kernelfunctions[m.kernel].(d², m.ϵ)

    # predict as linear combination
    ynew = M' * coefs
    return ynew
end


function partial_fit(m::RBFRegression, fitresults, X, y)
    if fitresults === nothing
        return fit(m, X, y; minv=true)
    end
    _, Xm, ym, Minv = fitresults
    if Minv === nothing
        throw(ArgumentError(
            "Expected `fitresults` to contain the inverse of the kernel matrix. Have \
            you run `fit(...; minv=true)`?"
        ))
    end

    # process inputs
    #  - X ∈ (n_samples, n_features)
    #  - y ∈ (n_samples, 1)
    Xp = _tomat(X)  # p for partial
    yp = _tomat(y)

    # compute distance of new data w.r.t. old points and itself
    kf = _kernelfunctions[m.kernel]
    Φ = kf.(pairwise(sqeuclideandist, Xm, Xp, dims=1), m.ϵ)
    ϕ = kf.(pairwise(sqeuclideandist, Xp, dims=1), m.ϵ)

    # compute new inverse of M via blockwise inversion
    c = inv(ϕ - Φ' * Minv * Φ)
    A = Minv + Minv * Φ * c * Φ' * Minv
    B = -Minv * Φ * c
    Minvnew = [A B; B' c]

    # compute new coefficients
    Xmnew = [Xm; Xp]
    ymnew = [ym; yp]
    coefsnew = Minvnew * ymnew
    return (coefsnew, Xmnew, ymnew, Minvnew), nothing, NamedTuple{}()
end


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


fitted_params(m::IDWRegression, _) = ()


function predict(m::IDWRegression, (Xm, ym), Xnew)
    # process input
    Xnewm = _tomat(Xnew)

    # create matrix of weights
    d² = pairwise(sqeuclideandist, Xm, Xnewm, dims=1)
    W = 1 ./ (d² .+ δ)
    if m.weighting == :expinversesquared
        W .*= exp.(-d²)
    end

    # predict as weighted average
    v = W ./ sum(W, dims=1)
    ynew = v' * ym
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
