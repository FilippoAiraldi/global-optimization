"""
Example of application of the GO nonmyopic algorithm. This example attempts to reproduce
Fig. 7 of [1], but with a different RBF kernel.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from math import ceil
from random import seed

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.multi_step_lookahead import warmstart_multistep
from botorch.optim import optimize_acqf
from torch import Tensor

from globopt import (
    GaussHermiteSampler,
    IdwAcquisitionFunction,
    Ms,
    Rbf,
    make_idw_acq_factory,
)
from globopt.problems import SimpleProblem

seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)  # with RBF regressor, float32 may not be enough
torch.set_default_device(torch.device("cpu"))
plt.style.use("bmh")


# instantiate problem and create starting training data
N_ITERS = 6
problem = SimpleProblem()
lb, ub = problem._bounds[0]
bounds = torch.as_tensor([[lb], [ub]])
train_X = torch.as_tensor([[-2.62, -1.2, 0.14, 1.1, 2.82]]).T
train_Y = problem(train_X)
eps, c1, c2 = 0.5, 1.0, 0.5
fantasies = [1, 1]
horizon = len(fantasies) + 1
n_restarts = 16 * horizon
raw_samples = 16 * 8 * horizon

# start regressor state and the previous full optimizer as None
rbf_state = full_opt = None

# auxiliary quantities for plotting
x_plot = torch.linspace(lb, ub, 100)
trajectories = torch.stack(
    torch.meshgrid(*(x_plot for _ in range(horizon)), indexing="ij"), axis=-1
)
trajectories[..., 1:] += torch.randn_like(trajectories[..., 1:]) * 1e-3
trajectories_ = trajectories.view(-1, horizon, 1)
x_plot = x_plot.view(-1, 1, 1)
history: list[tuple[Tensor, ...]] = []


# run optimization loop
for iteration in range(N_ITERS):
    # instantiate model and acquisition function
    mdl = Rbf(train_X, train_Y, eps, init_state=rbf_state)
    # remaining_horizon = min(horizon, N_ITERS - iteration)
    NMAF = Ms(
        model=mdl,
        fantasies_samplers=[GaussHermiteSampler(torch.Size([f])) for f in fantasies],
        valfunc_cls=IdwAcquisitionFunction,
        valfunc_argfactory=make_idw_acq_factory(c1, c2),
    )

    # minimize acquisition function
    q = NMAF.get_augmented_q_batch_size(1)
    if full_opt is not None:
        full_opt = warmstart_multistep(NMAF, bounds, n_restarts, raw_samples, full_opt)
    full_opt, tree_vals = optimize_acqf(
        NMAF,
        bounds,
        q,
        n_restarts,
        raw_samples,
        batch_initial_conditions=full_opt,
        return_best_only=False,
        return_full_tree=True,
        options={"seed": iteration, "maxfun": 15_000},
    )
    best_tree_idx = tree_vals.argmax()
    acq_opt = tree_vals[best_tree_idx]
    X_opt = NMAF.extract_candidates(full_opt[best_tree_idx])

    # evaluate objective function at the new point, and append it to training data
    Y_opt = problem(X_opt)
    train_X = torch.cat((train_X, X_opt))
    train_Y = torch.cat((train_Y, Y_opt))
    rbf_state = mdl.state
    prev_full_opt = full_opt

    # compute quantities for plotting purposes
    a_all_values = NMAF(trajectories_).view(trajectories.shape[:-1])
    a = a_all_values.amax(dim=tuple(range(1, a_all_values.ndim)))
    historic_item = (mdl(x_plot.transpose(0, 1))[0], a, X_opt, Y_opt, acq_opt)
    history.append(tuple(map(torch.squeeze, historic_item)))

# do plotting
n_cols = 3
n_rows = ceil(len(history) / n_cols)
axs = plt.subplots(
    n_rows,
    n_cols,
    constrained_layout=True,
    figsize=(2.5 * n_cols, 2 * n_rows),
    sharex=True,
    sharey=True,
)[1].flatten()
x_plot = x_plot.squeeze()
y_plot: Tensor = problem(x_plot).squeeze()
for i, (ax, historic_item) in enumerate(zip(axs, history)):
    regr_pred, acq_func, x_opt, y_opt, acq_opt = historic_item

    # plot true function, current sampled points, and regression prediction
    n = 5 + i
    X_so_far, Y_so_far = train_X[:n, 0], train_Y[:n, 0]
    ax.plot(x_plot, y_plot, label="$f(x)$", color="C0")
    ax.plot(X_so_far, Y_so_far, "o", color="C0", markersize=8)
    ax.plot(x_plot, regr_pred, label=r"$\hat{f}(x)$", color="C1")

    # plot acquisition function and its minimum
    ax_ = ax.twinx()
    ax_.plot(x_plot, acq_func, "--", lw=2.5, label="$a(x)$", color="C2")
    ax_.plot(x_opt, acq_opt, "*", markersize=13, color="C2")
    ax_.set_axis_off()
    ax_.set_ylim(-2.3, 0.8)
    ax.set_ylim(-0.5, 3.2)

    # plot next observation point
    ax.plot(x_opt, y_opt, "o", markersize=8, color="C4")

    # set axis limits and title
    best_y = min(Y_so_far.min().item(), y_opt.item())
    ax.set_xlim(lb, ub)
    ax.set_title(f"iter = {i + 1}, best cost = {best_y:.4f}", fontsize=9)
for j in range(i + 1, len(axs)):
    axs[j].set_axis_off()
plt.show()
