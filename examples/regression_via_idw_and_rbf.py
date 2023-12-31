"""
Example of regression with inverse distance weighting (IDW) and radial basis functions
(RBF) on a simple scalar function. This example attempts to reproduce Fig. 1 of [1].

These regression functionalities are used in the implementation of the myopic and
non-myopic solvers and are not intended to be used directly by the user.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from typing import Union

import matplotlib.pyplot as plt
import torch

from globopt import Idw, Rbf
from globopt.problems import SimpleProblem

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float32)  # with RBF regressor, float32 may not be enough
torch.set_default_device(torch.device("cpu"))
plt.style.use("bmh")


# create data points - X has shape (batch, n_samples, dim). Since we only have one
# batch of data, its dimension is 1
problem = SimpleProblem()
train_X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device="cpu").unsqueeze(-1)
train_Y = problem(train_X)

# fit regression models - only first n points for now
n = 3
mdls: list[Union[Idw, Rbf]] = [
    Idw(train_X[:n], train_Y[:n]),
    Rbf(train_X[:n], train_Y[:n], eps=0.5),
    Rbf(train_X[:n], train_Y[:n], eps=2.0),
]

# to partially fit new data, use the `fantasize` method, or manually pass the new
# dataset to the constructor, and only the newest data will be used
for i in range(len(mdls)):
    mdls[i] = mdls[i].fantasize(train_X[n:], sampler=lambda _: train_Y[n:])
    # mdl = mdls[i]
    # if isinstance(mdl, Idw):
    #     mdls[i] = Idw(train_X, train_Y)  # easier than RBFs
    # else:
    #     # for RBFs, pass also the previously fitted results
    #     mdls[i] = Rbf(train_X, train_Y, mdl.eps, mdl.svd_tol, (mdl.Minv, mdl.coeffs))

# predict values over all domain via the fitted models
X = torch.linspace(-3, 3, 1000).view(-1, 1, 1)
Y_hat = [mdl.posterior(X).mean for mdl in mdls]

# plot model predictions
_, ax = plt.subplots(constrained_layout=True, figsize=(7, 3))
ax.plot(train_X.squeeze(), train_Y.squeeze(), "o", markersize=9, color="C0")
ax.plot(X.squeeze(), problem(X).squeeze(), "--", label="f(x)", lw=3)
for mdl, Y_hat_ in zip(mdls, Y_hat):
    ax.plot(X.squeeze(), Y_hat_.squeeze(), label=mdl.__class__.__name__)
ax.set_xlabel("x")
ax.set_xlim(*problem._bounds[0])
ax.set_ylim(0, 2.5)
ax.legend()
plt.show()
