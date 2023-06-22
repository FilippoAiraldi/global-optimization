"""
Example of computation and minimization of the myopic acquisition function on a simple
scalar function. This example attempts to reproduce Fig. 3 and 6 of [1].

These functionalities are used in the implementation of the myopic and non-myopic
solvers and are not intended to be used directly by the user.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


import matplotlib.pyplot as plt
import numpy as np
from vpso import vpso

from globopt.core.problems import Simple1dProblem
from globopt.core.regression import Kernel, Rbf, RegressorType, fit, predict
from globopt.myopic.acquisition import (
    _idw_distance,
    _idw_variance,
    _idw_weighting,
    acquisition,
)

plt.style.use("bmh")

# define the function and its domain
lb, ub = -3, +3
f = Simple1dProblem.f

# create data points - X has shape (batch, n_samples, dim)
X = np.array([-2.61, -1.92, -0.63, 0.38, 2]).reshape(1, -1, 1)
y = f(X)

# create regressor and fit it
mdl: RegressorType = Rbf(Kernel.ThinPlateSpline, 0.01)
mdl = fit(mdl, X, y)

# predict values over all domain via fitted model
x = np.linspace(lb, ub, 1000).reshape(1, -1, 1)
y_hat = predict(mdl, x)

# compute acquisition function components (these methods should not be used directly)
dym = y.ptp(1, keepdims=True)  # span of observations
W = _idw_weighting(x, X, mdl.exp_weighting)
s = _idw_variance(y_hat, y, W)
z = _idw_distance(W)

# compute the overall acquisition function
a = acquisition(x, mdl, 1.0, 0.5, y_hat, dym)

# compute minimizer of acquisition function
res = vpso(
    func=lambda x: acquisition(x, mdl, 1.0, 0.5, None, None)[..., 0],
    lb=np.array([[lb]]),
    ub=np.array([[ub]]),
    seed=1909,
)

# create figure and flatten a bunch of arrays for plotting
_, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 5))
x, X, y, y_hat, s, z, a = (o.squeeze() for o in (x, X, y, y_hat, s, z, a))

# plot the function, the observations and the prediction in both axes
for ax in axs:
    c = ax.plot(x, f(x), label="$f(x)$")[0].get_color()
    ax.plot(X, y, "o", label=None, color=c)
    c = ax.plot(x, y_hat, label=None)[0].get_color()

# plot acquisition function components
axs[0].fill_between(
    x,
    y_hat - s,
    y_hat + s,
    label=r"$\hat{f}(x) \pm s(x)$ " + str(mdl),
    color=c,
    alpha=0.2,
)
axs[0].plot(x, z, label="$z(x)$")

# plot acquisition function and its minimizer
c = axs[1].plot(x, a, "--", lw=2.5, label="$a(x)$")[0].get_color()
axs[1].plot(
    res[0].item(),
    res[1].item(),
    "*",
    label=r"arg min $a(x)$",
    markersize=17,
    color=c,
)

# make plots look nice
for ax in axs:
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 2.5)
    ax.legend()
plt.show()
