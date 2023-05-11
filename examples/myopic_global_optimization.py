"""
Example of application of the GO myopic algorithm. This example attempts to reproduce
Fig. 7 of [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.normalization import NoNormalization

from globopt.core.regression import Array, Rbf, predict
from globopt.myopic.algorithm import GO, acquisition
from globopt.util.normalization import RangeNormalization

plt.style.use("bmh")


class Simple1DProblem(Problem):
    """Simple scalar problem to be minimized. Supports normalization."""

    def __init__(self, normalized: bool = True) -> None:
        if normalized:
            xl, xu = -1, 1
            self.normalization = RangeNormalization(-3, 3, xl, xu)
        else:
            xl, xu = -3, 3
            self.normalization = NoNormalization()
        super().__init__(n_var=1, n_obj=1, xl=xl, xu=xu, type_var=float)

    def _evaluate(self, x: Array, out: dict[str, Any], *_, **__) -> None:
        x = self.normalization.backward(x)
        out["F"] = (
            (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
            + x**2 / 12
            + x / 10
        )

    def _calc_pareto_front(self) -> float:
        """Returns the global minimum of the problem."""
        return 0.279504

    def _calc_pareto_set(self) -> float:
        """Returns the global minimizer of the problem."""
        return self.normalization.forward(-0.959769).item()


# instantiate problem and create starting training data
problem = Simple1DProblem(normalized=False)
x0 = problem.normalization.forward([-2.62, -1.2, 0.14, 1.1, 2.82])

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
y = problem.evaluate(x)
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
    c = ax.plot(x, y, label="$f(x)$")[0].get_color()
    ax.plot(Xm, ym, "o", color=c, markersize=8)

    # plot current regression model's prediction
    mdl = algo.regression
    y_hat = predict(mdl, x[np.newaxis])
    ax.plot(x, y_hat[0], label=r"$\hat{f}(x)$")

    # plot the acquisition function and its minimum, or or the best point found if the
    # algorithm has terminated
    a = acquisition(x[None], mdl, y_hat, **algo.acquisition_fun_kwargs)[0]
    c = ax.plot(x, a, "--", lw=2.5, label="$a(x)$")[0].get_color()
    if i < len(result.history) - 1:
        acq_min = result.history[i + 1].acquisition_min_res.opt.item()
        ax.plot(acq_min.X, acq_min.F, "*", markersize=17, color=c)
    else:
        ax.plot(*algo.opt.get("X", "F"), "*", markersize=17, color="k")

    # set axis limits and title
    ax.set_xlim(*problem.bounds())
    ax.set_ylim(0, 2.5)
    ax.set_title(f"iter = {i + 1}, best cost = {ym.min():.4f}", fontsize=9)
    if i == 0:
        ax.legend()
for j in range(i + 1, len(axs)):
    axs[j].set_axis_off()
plt.show()
