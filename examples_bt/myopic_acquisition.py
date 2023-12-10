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
import torch
from botorch.optim import optimize_acqf

from globopt.myopic_acquisitions import (
    GoMyopicAcquisitionFunction,
    _idw_distance,
    acquisition_function,
)
from globopt.problems import SimpleProblem
from globopt.regression import Rbf

plt.style.use("bmh")

# define the function and its domain
problem = SimpleProblem()
lb, ub = problem._bounds[0]

# create data points - X has shape (batch, n_samples, dim)
dv = "cpu"
train_X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device=dv).reshape(1, -1, 1)
train_Y = problem(train_X)

# create regressor and fit it
mdl = Rbf(train_X, train_Y, 0.5)

# predict the (normal) posterior over all domain via fitted model
X = torch.linspace(lb, ub, 1000).reshape(1, -1, 1)
posterior = mdl(X)
y_hat = posterior.mean

# compute acquisition function by components
s = posterior.scale  # a.k.a., idw scale
z = _idw_distance(mdl.W_sum_recipr)

# compute the overall acquisition function
a = acquisition_function(y_hat, s, train_Y, mdl.W_sum_recipr, 1.0, 0.5).squeeze()

# compute minimizer of acquisition function
x_opt, a_opt = optimize_acqf(
    acq_function=GoMyopicAcquisitionFunction(mdl, 1.0, 0.5),
    bounds=torch.as_tensor([[lb], [ub]]).reshape(2, -1),
    q=1,
    num_restarts=20,
    raw_samples=100,
)

# plot the function, the observations and the prediction in both axes
_, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 5))
X = X.squeeze()
for ax in axs:
    c = ax.plot(X, problem(X), label="$f(x)$")[0].get_color()
    ax.plot(train_X.squeeze(), train_Y.squeeze(), "o", label=None, color=c)
    c = ax.plot(X, y_hat.squeeze(), label=None)[0].get_color()

# plot acquisition function components
axs[0].fill_between(
    X,
    (y_hat - s).squeeze(),
    (y_hat + s).squeeze(),
    label=r"$\hat{f}(x) \pm s(x)$",
    color=c,
    alpha=0.2,
)
axs[0].plot(X, z.squeeze(), label="$z(x)$")

# plot acquisition function and its minimizer
c = axs[1].plot(X, a - a.min(), "--", lw=2.5, label="$a(x)$")[0].get_color()
axs[1].plot(
    x_opt.squeeze(),
    a_opt - a.min(),
    "*",
    label=r"arg max $a(x)$",
    markersize=17,
    color=c,
)

# make plots look nice
for ax in axs:
    ax.set_xlim(lb, ub)
    ax.set_ylim(0, 2.5)
    ax.legend()
plt.show()
