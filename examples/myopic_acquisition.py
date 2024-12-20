"""
Example of computation and minimization of myopic acquisition functions on a simple
scalar function. These include the vanilla myopic acquisition function from [1], and its
stochastic version (i.e., with expectation over the posterior distribution) computed via
both Monte Carlo (MC) and Gauss-Hermite (GH) quadrature. This example is inspired by
Fig. 3 and 6 of [1].

Most of these functionalities are used in the implementation of the myopic and
non-myopic solvers and are not intended to be used directly by the user.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


import matplotlib.pyplot as plt
import torch
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler

from globopt import (
    GaussHermiteSampler,
    IdwAcquisitionFunction,
    Rbf,
    qIdwAcquisitionFunction,
)
from globopt.myopic_acquisitions import _idw_distance, idw_acquisition_function
from globopt.problems import SimpleProblem

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
X = torch.linspace(lb, ub, 1000).view(1, -1, 1)
y_hat, s, W_sum_recipr, _ = mdl(X)

# compute the overall myopic vanilla acquisition function, component by component
c1 = torch.scalar_tensor(1.0)
c2 = torch.scalar_tensor(0.5)
y_span = train_Y.amax(-2, keepdim=True) - train_Y.amin(-2, keepdim=True)
z = _idw_distance(W_sum_recipr)
a = idw_acquisition_function(y_hat, s, y_span, W_sum_recipr, c1, c2).squeeze()

# compute minimizer of analytic myopic acquisition function
x_opt, a_opt = optimize_acqf(
    acq_function=IdwAcquisitionFunction(mdl, c1, c2),
    bounds=bounds,
    q=1,  # mc iterations - not supported for the analytical acquisition function
    num_restarts=16,  # number of optimization restarts
    raw_samples=32,  # initial samples to start the first `num_restarts` points
    options={"seed": 0},
)

# for the monte carlo version, we can directly use the forward method
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2**5]), seed=0)
MCAF = qIdwAcquisitionFunction(mdl, c1, c2, sampler)
a_mc = MCAF(X.view(-1, 1, 1))
x_opt_mc, a_opt_mc = optimize_acqf(
    acq_function=MCAF,
    bounds=bounds,
    q=1,
    num_restarts=64,
    raw_samples=128,
    options={"seed": 0},
)

# instead of the quasi-monte carlo approach, we can also use a version that approximates
# the expected value via Gauss-Hermite quadrature
sampler = GaussHermiteSampler(sample_shape=torch.Size([2**2]))
GHAF = qIdwAcquisitionFunction(mdl, c1, c2, sampler)
a_exp = GHAF(X.view(-1, 1, 1))
x_opt_exp, a_opt_exp = optimize_acqf(
    acq_function=GHAF,
    bounds=bounds,
    q=1,
    num_restarts=16,
    raw_samples=32,
    options={"seed": 0},
)

# plot
_, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 2.5))
X = X.squeeze()
ax.plot(X, problem(X), label="Unknown objective $f(x)$", color="C0")
ax.plot(train_X.squeeze(), train_Y.squeeze(), "o", label=None, color="C0")
ax.plot(X, y_hat.squeeze(), label=r"Surrogate model $\hat{f}(x) \pm s(x)$", color="C1")
ax.fill_between(X, (y_hat - s).squeeze(), (y_hat + s).squeeze(), color="C1", alpha=0.2)
ax.plot(X, z.squeeze(), label="Distance function $z(x)$", color="C2")
names = [
    r"Vanilla acquisition $\Lambda$",
    r"Stochastic acquisition $\Lambda^\text{s}$ (MC)",
    r"Stochastic acquisition $\Lambda^\text{s}$ (GH)",
]
data = [(a, x_opt, a_opt), (a_mc, x_opt_mc, a_opt_mc), (a_exp, x_opt_exp, a_opt_exp)]
for i, (name, (a_, x_opt_, a_opt_)) in enumerate(zip(names, data)):
    c = f"C{i + 3}"
    a_min = a_.amin()
    ax.plot(X, (a_ - a_min).squeeze(), "--", lw=1, label=name, color=c)
    ax.plot(
        x_opt_.squeeze(),
        (a_opt_ - a_min).squeeze(),
        "*",
        label=None,  # r"$\arg \max a(x)$",
        markersize=17,
        color=c,
    )
ax.set_xlim(lb, ub)
ax.set_ylim(0, 2.5)
ax.legend()
plt.show()
