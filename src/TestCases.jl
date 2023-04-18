module TestCases
export ndims, TestCase, BraninHoo, bemporad2020


mutable struct TestCase{T}
    f::Function
    bounds::Array{Float64, T}
end

ndims(tc::TestCase)::Integer = Base.ndims(tc.bounds)

@inline _scale(x, lb, ub) = 2 * (x - lb) / (ub - lb) - 1
@inline _unscale(x, lb, ub) = (x + 1) * (ub - lb) / 2 + lb

raw"""
    braninhoo

Test case for the
[Branin-Hoo](http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano/branin.html) function.

...math::
    f(x) = a(x_2 - b x_1^2 + c x_1 - r)^2 + s(1 - t) cos(x_1) + s

# Description
Dimensions: 2
The Branin, or Branin-Hoo, function has three global minima. The recommended values of
`a`, `b`, `c`, `r`, `s` and `t` are: `a = 1`, `b = 5.1 / (4π^2)`, `c = 5 / π`, `r = 6`,
`s = 10` and `t = 1 / (8π)`.

# Input Domain:
This function is usually evaluated on the box `x_1 ∈ [-5, 10]`, `x_2 ∈ [0, 15]`, but it
is scaled to ranges `[-1, 1]`.

# Global Minimum (unscaled):
.. math::
    f(x_\star) = 0.397887, at (x_1, x_2) = (-\pi, 12.275), (12.275, 2.275),
    (9.42478, 2.475)
"""
function braninhoo(
    x, y; a=1, b=5.1 / (4π^2), c=5 / π, r=6, s=10, t=1 / (8π), scaled::Bool = true
)
    if scaled
        x = _unscale(x, -5, +10)
        y = _unscale(y, 0, +15)
    end
    return a * (y - b * x^2 + c * x - r)^2 + s * (1 - t) * cos(x) + s
end


raw"""
    bemporad2020

Test case for the 1D function from [1].

...math::
    f(x) = (1 + x sin(2x) cos(3x) / (1 + x^2))^2 + x^2 / 10 + x / 10

# Description
Dimensions: 1

# Input Domain:
This function is usually evaluated on the range `x ∈ [-3, 3]`, but it is scaled to range
`[-1, 1]`.

# Global Minimum (unscaled):
.. math::
    f(x_\star) = 0.27945, at x = -0.9599.

# References
[1] Bemporad, A. Global optimization via inverse distance weighting and radial basis
    functions. Comput Optim Appl 77, 571–595 (2020).
    https://doi-org.tudelft.idm.oclc.org/10.1007/s10589-020-00215-w
"""
function bemporad2020(x; scaled::Bool = true)
    if scaled
        x = _unscale(x, -3, +3)
    end
    return (1 + x * sin(2x) * cos(3x) / (1 + x^2))^2 + x^2 / 10 + x / 10
end

end
