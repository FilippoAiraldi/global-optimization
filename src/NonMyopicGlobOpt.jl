"""
This package exports
* regressor types: `RBFRegression`, `IDWRegression`
"""

module NonMyopicGlobOpt

include("regression.jl")
include("util.jl")

export RBFRegression, IDWRegression, fit, partial_fit, predict, fitted_params, logrange

end
