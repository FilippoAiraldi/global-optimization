# https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/

module Regressors
export RBFRegressor, _kernelfunctions


import MLJModelInterface
const MMI = MLJModelInterface
import MLJBase
using Distances: pairwise, SqEuclidean


@inline _tomat(X) = begin
    if ndims(X) == 1
        X = reshape(X, :, 1)
    end
    return MLJBase.matrix(X)
end


_kernelfunctions::Dict{Symbol,Function} = Dict(
    :inversequadratic => (d2, eps) -> 1 / (1 + eps^2 * d2),
    :multiquadric => (d2, eps) -> sqrt(1 + eps^2 * d2),
    :linear => (d2, eps) -> eps * sqrt(d2),
    :gaussian => (d2, eps) -> exp(-eps^2 * d2),
    :thinplatespline => (d2, eps) -> eps^2 * d2 * log(eps * sqrt(d2) .+ 1e-6),
    :inversemultiquadric => (d2, eps) -> 1 / sqrt(1 + eps^2 * d2),
)


"""
Linear regression with radial basis function (RBF) kernels.

# Fields
- kernel: type of RBF function to use. One of `:inversequadratic`, `:multiquadric`,
  `:linear`, `:gaussian`, `:thinplatespline`, `:inversemultiquadric`.
- eps: parameter to scale distances.
"""
MMI.@mlj_model mutable struct RBFRegressor <: MMI.Deterministic
    kernel::Symbol = :inversequadratic::(_ in (
        :inversequadratic,
        :multiquadric,
        :linear,
        :gaussian,
        :thinplatespline,
        :inversemultiquadric,
    ))
    eps::Float64 = 1e-3::(_ >= 0)
end


function MLJBase.fit(m::RBFRegressor, X, y)
    # process inputs
    #  - X ∈ (n_samples, n_features)
    #  - y ∈ (n_samples, 1)
    Xm = _tomat(X)
    ym = _tomat(y)

    # create matrix of kernel evaluations
    d = pairwise(SqEuclidean(), Xm, dims=1)
    M = _kernelfunctions[m.kernel].(d, m.eps)

    # compute coefficients
    coefs = M \ ym
    return (coefs, Xm), nothing, NamedTuple{}()
end


MLJBase.fitted_params(m::RBFRegressor, (coefs, _)) = (coefs=coefs,)


function MLJBase.predict(m::RBFRegressor, (coefs, Xm), Xnew)
    # process input
    Xnewm = _tomat(Xnew)

    # create matrix of kernel evaluations
    d = pairwise(SqEuclidean(), Xm, Xnewm, dims=1)
    M = _kernelfunctions[m.kernel].(d, m.eps)

    # predict as linear combination
    ynew = M' * coefs
    return ynew
end

# TODO: partial_fit
end
