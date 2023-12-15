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
    MyopicAcquisitionFunction,
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
train_X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device=dv).view(-1, 1)
train_Y = problem(train_X)

# create regressor and fit it
mdl = Rbf(train_X, train_Y, 0.5)

# predict the (normal) posterior over all domain via fitted model
X = torch.linspace(lb, ub, 1000).view(1, -1, 1)
y_hat, s, W_sum_recipr = mdl(X)

# compute acquisition function by components
z = _idw_distance(W_sum_recipr)

# compute the overall acquisition function
c1 = 1.0
c2 = 0.5
y_span = train_Y.amax(-2, keepdim=True) - train_Y.amin(-2, keepdim=True)
a = acquisition_function(y_hat, s, y_span, W_sum_recipr, c1, c2).squeeze()

# compute minimizer of analytic myopic acquisition function
myopic_analytic_optimizer, myopic_analitic_opt = optimize_acqf(
    acq_function=MyopicAcquisitionFunction(mdl, c1, c2),
    bounds=torch.as_tensor([[lb], [ub]]),
    q=1,  # mc iterations - not supported for the analytical acquisition function
    num_restarts=10,  # number of optimization restarts
    raw_samples=20,  # initial samples to start the first `num_restarts` points
    options={"seed": 0},
)

# # compute minimizer of MC myopic acquisition function
# myopic_mc_optimizer, myopic_mc_opt = optimize_acqf(
#     acq_function=qMcMyopicAcquisitionFunction(mdl, c1, c2),
#     bounds=torch.as_tensor([[lb], [ub]]),
#     q=2**10,
#     num_restarts=10,
#     raw_samples=20,
#     options={"seed": 0},
# )

# plot
_, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 2.5))
X = X.squeeze()
ax.plot(X, problem(X), label="$f(x)$", color="C0")
ax.plot(train_X.squeeze(), train_Y.squeeze(), "o", label=None, color="C0")
ax.plot(X, y_hat.squeeze(), label=None, color="C1")
ax.fill_between(
    X,
    (y_hat - s).squeeze(),
    (y_hat + s).squeeze(),
    label=r"$\hat{f}(x) \pm s(x)$",
    color="C1",
    alpha=0.2,
)
ax.plot(X, z.squeeze(), label="$z(x)$", color="C2")
ax.plot(X, a - a.min(), "--", lw=2.5, label="Analitycal $a(x)$", color="C3")
ax.plot(
    myopic_analytic_optimizer.squeeze(),
    myopic_analitic_opt - a.min(),
    "*",
    label=None,  # r"$\arg \max a(x)$",
    markersize=17,
    color="C3",
)
ax.set_xlim(lb, ub)
ax.set_ylim(0, 2.5)
ax.legend()
plt.show()
