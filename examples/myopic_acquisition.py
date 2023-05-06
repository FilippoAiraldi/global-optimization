"""
Example of computation and minimization of the myopic acquisition function on a simple
scalar function. This example attempts to reproduce Fig. 3 and 6 of [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # no need for jit in this example

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from globopt.core.regression import Rbf, RegressorType, fit, predict
from globopt.myopic.acquisition import (
    _idw_distance,
    _idw_variance,
    acquisition,
    idw_weighting,
)

plt.style.use("bmh")


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


# create data points
X = np.array([[-2.61, -1.92, -0.63, 0.38, 2]]).T
y = f(X)

# create regressor and fit it
mdl: RegressorType = Rbf("thinplatespline", 0.01)
mdl = fit(mdl, X, y)

# predict values over all domain via fitted model
x = np.linspace(-3, 3, 1000).reshape(1, -1, 1)
y_hat = predict(mdl, x)

# compute acquisition function components (these methods should not be used directly)
X = np.expand_dims(X, 0)  # add batch dimension
y = np.expand_dims(y, 0)
dym = y.max() - y.min()  # span of observations
W = idw_weighting(x, X, mdl.exp_weighting)
s = _idw_variance(y_hat, y, W)
z = _idw_distance(W)

# compute the overall acquisition function
a = acquisition(x, mdl, y_hat, dym, c1=1, c2=0.5)

# compute minimizer of acquisition function
algorithm = PSO()
problem = FunctionalProblem(
    n_var=1,
    objs=lambda x: acquisition(x, mdl, c1=1, c2=0.5),
    xl=-3,
    xu=3,
    elementwise=False,  # enables vectorized evaluation of acquisition function
)
res = minimize(problem, algorithm, verbose=True, seed=1)

# create figure and flatten a bunch of arrays for plotting
_, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(7, 3))
x = x.flatten()
fx = f(x)
a = a.flatten()
y_hat = y_hat.flatten()
s = s.flatten()
z = z.flatten()
X, y = X[0], y[0]

# plot acquisition function components
line = axs[0].plot(x, fx, label="$f(x)$")[0]
axs[0].plot(X, y, "o", label=None, color=line.get_color())
line = axs[0].plot(x, y_hat, label=None)[0]
axs[0].fill_between(
    x,
    y_hat - s,
    y_hat + s,
    label=r"$\hat{f}(x) \pm s(x)$ " + str(mdl),
    color=line.get_color(),
    alpha=0.2,
)
axs[0].plot(x, z, label="$z(x)$")

# plot acquisition function and its minimizer
line = axs[1].plot(x, fx, label="$f(x)$")[0]
axs[1].plot(X, y, "o", label=None, color=line.get_color())
axs[1].plot(x, y_hat)
line = axs[1].plot(x, a, label="$a(x)$")[0]
axs[1].plot(
    res.X.item(),
    res.F.item(),
    "*",
    label=r"arg min $a(x)$",
    markersize=14,
    color=line.get_color(),
)

# ax.set_xlabel("x")
for ax in axs:
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 2.5)
    ax.legend()
plt.show()
