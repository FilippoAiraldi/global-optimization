"""
Example of application of the GO myopic algorithm. This example attempts to reproduce
Fig. 7 of [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


import matplotlib.pyplot as plt
import numpy as np
from pymoo.optimize import minimize

from globopt.core.problems import Simple1DProblem
from globopt.core.regression import Rbf, predict
from globopt.myopic.algorithm import GO, acquisition

plt.style.use("bmh")


# instantiate problem and create starting training data
problem = Simple1DProblem()
x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]

# instantiate algorithm and then run the optimization
algorithm = GO(
    regression=Rbf("thinplatespline", 0.01),
    init_points=x0,
    acquisition_fun_kwargs={"c1": 1, "c2": 0.5},
)
result = minimize(
    problem,
    algorithm,
    termination=("n_iter", 6),
    verbose=True,
    seed=1,
    save_history=True,
)

# plot the results
x = np.linspace(*problem.bounds(), 500).reshape(-1, 1)  # type: ignore[call-overload]
y = problem.evaluate(x).reshape(-1)
n_cols = 4
n_rows = int(np.ceil(len(result.history) / n_cols))
_, axs = plt.subplots(
    n_rows, n_cols, constrained_layout=True, figsize=(2.5 * n_cols, 2 * n_rows)
)
axs = axs.flatten()
for i, (ax, algo) in enumerate(zip(axs, result.history)):
    # plot true function and current sampled points
    Xm = algo.pop.get("X").reshape(-1, 1)
    ym = algo.pop.get("F").reshape(-1)
    c = ax.plot(x.reshape(-1), y, label="$f(x)$")[0].get_color()
    ax.plot(Xm, ym, "o", color=c, markersize=8)

    # plot current regression model's prediction
    mdl = algo.regression
    y_hat = predict(mdl, x)
    ax.plot(x.reshape(-1), y_hat, label=r"$\hat{f}(x)$")

    # plot the acquisition function and its minimum, or the best point found if the
    # algorithm has terminated
    if i < len(result.history) - 1:
        a = acquisition(x, mdl, y_hat, **algo.acquisition_fun_kwargs)
        acq_min = result.history[i + 1].acquisition_min_res.opt.item()

        ax_ = ax.twinx()
        line = ax_.plot(x.reshape(-1), a, "--", lw=2.5, label="$a(x)$", color="C2")
        ax_.plot(acq_min.X, acq_min.F, "*", markersize=13, color=line[0].get_color())
        ax_.set_axis_off()
        ylim = ax_.get_ylim()
        ax_.set_ylim(ylim[0] - 0.1, ylim[1] + np.diff(ylim) * 0.7)
    else:
        ax.plot(*algo.opt.get("X", "F"), "*", markersize=17, color="C4")

    # set axis limits and title
    ax.set_xlim(*problem.bounds())
    ylim = (0.1, 2.4)
    ax.set_ylim(ylim[0] - np.diff(ylim) * 0.1, ylim[1])
    ax.set_title(f"iter = {i + 1}, best cost = {ym.min():.4f}", fontsize=9)
for j in range(i + 1, len(axs)):
    axs[j].set_axis_off()
plt.show()
