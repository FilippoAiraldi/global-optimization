# https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/

module Regression


import MLJModelInterface
const MMI = MLJModelInterface
import MLJBase: matrix, fit, predict, fitted_params
using Distances: pairwise, SqEuclidean


export RBFRegression, IDWRegression, fit, predict, fitted_params


const δ = 1e-6  # small number to avoid nans

@inline _tomat(X) = begin
    if ndims(X) == 1
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
Linear regression with radial basis function (RBF) kernels.

# Fields
- kernel: type of RBF function to use. One of `:inversequadratic`, `:multiquadric`,
  `:linear`, `:gaussian`, `:thinplatespline`, `:inversemultiquadric`.
- eps: parameter to scale distances.
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
    d² = pairwise(SqEuclidean(), Xm, Xnewm, dims=1)
    M = _kernelfunctions[m.kernel].(d², m.ϵ)

    # predict as linear combination
    ynew = M' * coefs
    return ynew
end


# TODO: partial_fit for RBFRegression


"""
Linear regression with inverse distance weighting (IDW).
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


fitted_params(m::IDWRegression, (_, _)) = ()


function predict(m::IDWRegression, (Xm, ym), Xnew)
    # process input
    Xnewm = _tomat(Xnew)

    # create matrix of weights
    d = pairwise(SqEuclidean(), Xm, Xnewm, dims=1)

    # implement cases when x==x_i or x==x_j

    # W = 1 ./ d
    # if m.weighting == :expinversesquared
    #     W .*= exp.(-d)
    # end
end


# TODO: partial_fit for IDWRegression

end
