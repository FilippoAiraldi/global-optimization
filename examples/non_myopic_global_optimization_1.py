"""
Example of application of the GO non-myopic algorithm and comparison of it with its
myopic counterpart. The example is taken from Fig. 7
of [1].

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

from globopt.core.regression import Array, Rbf, predict
from globopt.myopic.algorithm import GO
from globopt.nonmyopic.algorithm import NonMyopicGO
from globopt.nonmyopic.acquisition import optimal_acquisition

plt.style.use("bmh")


class Simple1DProblem(Problem):
    """Simple scalar problem to be minimized."""

    def __init__(self) -> None:
        super().__init__(n_var=1, n_obj=1, xl=-3, xu=+3, type_var=float)

    def _evaluate(self, x: Array, out: dict[str, Any], *_, **__) -> None:
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
        return -0.959769


# instantiate problem and create starting training data
problem = Simple1DProblem()
x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]

# instantiate myopic and non-myopic algorithms and then run them on the problem
kwargs = {
    "regression": Rbf("thinplatespline", 0.01),
    "init_points": x0,
    "acquisition_fun_kwargs": {"c1": 1, "c2": 0.5},
}
algorithms = (
    GO(**kwargs),  # type: ignore[arg-type]
    NonMyopicGO(**kwargs, horizon=2, discount=1.0),  # type: ignore[arg-type]
)
ITERS = 4
results = (
    minimize(
        problem,
        algorithm,
        termination=("n_iter", ITERS),
        verbose=True,
        seed=1,
        save_history=True,
    )
    for algorithm in algorithms
)

# plot myopic and non-myopic results results
x = np.linspace(*problem.bounds(), 60).reshape(-1, 1)  # type: ignore[call-overload]
y = problem.evaluate(x)
n_cols = min(ITERS, 4)
n_rows = int(np.ceil(ITERS / n_cols)) * 2
_, all_axs = plt.subplots(
    n_rows, n_cols, constrained_layout=True, figsize=(2.5 * n_cols, 2 * n_rows)
)
for result, ylbl, axs in zip(results, ("Myopic", "Non-myopic"), np.split(all_axs, 2)):
    axs = axs.flatten()
    for i, (ax, algo) in enumerate(zip(axs, result.history)):
        # plot true function and current sampled points
        Xm = algo.pop.get("X").reshape(-1, 1)
        ym = algo.pop.get("F").reshape(-1)
        c = ax.plot(x, y, label="$f(x)$")[0].get_color()
        ax.plot(Xm, ym, "o", color=c, markersize=8)

        # plot current regression model's prediction
        mdl = algo.regression
        y_hat = predict(mdl, x[np.newaxis])[0]
        ax.plot(x, y_hat, label=r"$\hat{f}(x)$")

        # plot the optimal acquisition function and its minimum, or the best point found
        # if the algorithm has terminated
        if i < len(result.history) - 1:
            h = getattr(algo, "horizon", 1)
            a = optimal_acquisition(
                x, mdl, h, **algo.acquisition_fun_kwargs, verbosity=10
            )
            acq_min = result.history[i + 1].acquisition_min_res.opt.item()
            p = acq_min.X[: problem.n_var], acq_min.F[: problem.n_var]

            ax_ = ax.twinx()
            c = ax_.plot(x[:, 0], a, "--", label="$a(x)$", color="C2")[0].get_color()
            ax_.plot(*p, "*", markersize=13, color=c)
            ax_.set_axis_off()
            ylim = ax_.get_ylim()
            ax_.set_ylim(ylim[0] - 0.1, ylim[1] + np.diff(ylim) * 0.7)
        else:
            ax.plot(*algo.opt.get("X", "F"), "*", markersize=17, color="C4")

        # set axis limits and title
        ax.set_xlim(*problem.bounds())
        ylim = (0.1, 2.4)
        ax.set_ylim(ylim[0] - np.diff(ylim) * 0.1, ylim[1])
        ax.set_title(f"iteration {i + 1}, best cost = {ym.min():.4f}", fontsize=9)
        if i == 0:
            ax.set_ylabel(ylbl, fontsize=9)
    for j in range(i + 1, axs.size):
        axs[j].set_axis_off()
plt.show()
