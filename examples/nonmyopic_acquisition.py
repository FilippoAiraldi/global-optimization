"""
Example of computation and minimization of the myopic acquisition function on a simple
scalar function.

Most of these functionalities are used in the implementation of the myopic and
non-myopic solvers and are not intended to be used directly by the user.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from random import seed

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.optim import optimize_acqf

from globopt import (
    GaussHermiteSampler,
    Ms,
    Rbf,
    make_idw_acq_factory,
    qIdwAcquisitionFunction,
)
from globopt.problems import SimpleProblem
from globopt.regression import Rbf

seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)  # with RBF regressor, float32 may not be enough
torch.set_default_device(torch.device("cpu"))
plt.style.use("bmh")

# define the evaluation function and its domain
problem = SimpleProblem()
lb, ub = problem._bounds[0]
bounds = torch.as_tensor([[lb], [ub]])

# create data points
train_X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2]).view(-1, 1)
train_Y = problem(train_X)

# create regressor and fit it
mdl = Rbf(train_X, train_Y, 0.5)

# predict the posterior over all domain via fitted model
X = torch.linspace(lb, ub, 1000)
y_hat, s = (o.cpu() for o in mdl(X.view(1, -1, 1))[:2])

# some default values
c1 = torch.scalar_tensor(1.0)
c2 = torch.scalar_tensor(0.5)
n_restarts = 32
raw_samples = 32 * 8
fantasies = [1, 1]
horizon = len(fantasies) + 1

# compute the nonmyopic acquisition function - this is trickier than the myopic one
# because now the acquisition is a function of trajectories of query points, not just
# the poins themselves. In other words, the input to the `forward` method of the
# acquisition function is a tensor of shape n x horizon x d, where n is the number of
# query points, horizon is the number of steps in the lookahead, and d is the dimension
# of the input space.
# Therefore, to plot it, we generate all possible combinations of trajectories,
# evaluate them, and plot the best one.
acqfun = Ms(
    model=mdl,
    fantasies_samplers=[GaussHermiteSampler(torch.Size([f])) for f in fantasies],
    valfunc_cls=qIdwAcquisitionFunction,
    valfunc_argfactory=make_idw_acq_factory(c1, c2),
    valfunc_sampler=GaussHermiteSampler(torch.Size([16])),
)

# generate all possible trajectories - add small perturbation to avoid numerical issues
X_subset = X[::10]  # reduce the number of points for plotting
trajectories = torch.stack(
    torch.meshgrid(*(X_subset for _ in range(horizon)), indexing="ij"), axis=-1
)
trajectories[..., 1:] += torch.randn_like(trajectories[..., 1:]) * 1e-3
a_all_values = acqfun(trajectories.view(-1, horizon, 1)).view(trajectories.shape[:-1])
a = a_all_values.amax(dim=tuple(range(1, a_all_values.ndim)))

# compute the minimizer of the nonmyopic acquisition function
x_opt, a_opt = optimize_acqf(
    acq_function=acqfun,
    bounds=bounds,
    q=acqfun.get_augmented_q_batch_size(1),
    num_restarts=n_restarts,
    raw_samples=raw_samples,
    options={"seed": 0, "maxfun": 10_000},
)

# plot
_, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 2.5))
X = X.cpu().squeeze()
X_subset = X_subset.cpu().squeeze()
ax.plot(X, problem(X), label="Unknown objective $f(x)$", color="C0")
ax.plot(train_X.cpu().squeeze(), train_Y.cpu().squeeze(), "o", label=None, color="C0")
ax.plot(X, y_hat.squeeze(), label=r"Surrogate model $\hat{f}(x) \pm s(x)$", color="C1")
ax.fill_between(X, (y_hat - s).squeeze(), (y_hat + s).squeeze(), color="C1", alpha=0.2)
names = [
    r"Vanilla acquisition $\Lambda$",
]
data = [(a, x_opt, a_opt)]
for i, (name, (a_, x_opt_, a_opt_)) in enumerate(zip(names, data)):
    c = f"C{i + 3}"
    a_ = a_.cpu().squeeze()
    x_opt_ = x_opt_.cpu().squeeze()
    a_opt_ = a_opt_.cpu().squeeze()
    a_min = a_.amin()
    ax.plot(X_subset.cpu(), a_ - a_min, "--", lw=1, label=name, color=c)
    ax.plot(x_opt_, a_opt_ - a_min, "*", markersize=17, color=c)
ax.set_xlim(lb, ub)
ax.set_ylim(0, 2.5)
ax.legend()
plt.show()
