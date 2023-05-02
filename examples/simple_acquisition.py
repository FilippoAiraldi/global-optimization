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

from globopt.core.regression import RBFRegression
from globopt.myopic.acquisition import (
    acquisition,
    idw_distance,
    idw_variance,
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
y = f(X).flatten()

# create regressor and fit it
mdl = RBFRegression("thinplatespline", 0.01)
mdl.fit(X, y)

# predict values over all domain via fitted model
x = np.linspace(-3, 3, 1000).reshape(-1, 1)
y_hat = mdl.predict(x)

# compute acquisition function components
dym = y.max() - y.min()  # span of observations
W = idw_weighting(x, X)
s = idw_variance(y_hat, y, W)
z = idw_distance(W)
a = acquisition(x, y_hat, X, y, dym, 1, 0.5)


# compute minimizer of acquisition function
algorithm = PSO()
problem = FunctionalProblem(
    n_var=1,
    objs=lambda x: acquisition(x, mdl.predict(x), X, y, dym, 1, 0.5),
    xl=-3,
    xu=3,
    elementwise=False,  # enables vectorized evaluation of acquisition function
)
res = minimize(problem, algorithm, verbose=True, seed=1)

# create figure
_, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(7, 3))

# plot acquisition function components
x = x.flatten()
fx = f(x)
axs[0].plot(x, fx, label="f(x)")
line = axs[0].plot(x, y_hat, label=None)[0]
axs[0].plot(X, y, "o", label=None, color=line.get_color())
axs[0].fill_between(
    x,
    y_hat - s,
    y_hat + s,
    label=f"f_hat(x) +/- s(x) ({mdl})",
    color=line.get_color(),
    alpha=0.2,
)
axs[0].plot(x, z, label="z(x)")

# plot acquisition function and its minimizer
axs[1].plot(x, fx, label="f(x)")
line = axs[1].plot(x, y_hat, label=str(mdl))[0]
axs[1].plot(X, y, "o", label=None, color=line.get_color())
line = axs[1].plot(x, a, label="a(x)")[0]
axs[1].plot(
    res.X.item(),
    res.F.item(),
    "D",
    label="arg min a(x)",
    markersize=7,
    color=line.get_color(),
)

# ax.set_xlabel("x")
for ax in axs:
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 2.5)
    ax.legend()
plt.show()
