"""
Another example comparing the myopic and non-myopic GO algorithms.
"""


from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from globopt.benchmarking.optimal_acquisition import optimal_acquisition
from globopt.core.regression import Array, Idw, predict
from globopt.myopic.algorithm import GO
from globopt.nonmyopic.algorithm import NonMyopicGO

plt.style.use("bmh")


class Simple1DProblem(Problem):
    """Simple scalar problem to be minimized."""

    def __init__(self) -> None:
        super().__init__(n_var=1, n_obj=1, xl=0, xu=1, type_var=float)

    def _evaluate(self, x: Array, out: dict[str, Any], *_, **__) -> None:
        out["F"] = x + np.sin(4.5 * np.pi * x)

    def _calc_pareto_front(self) -> float:
        """Returns the global minimum of the problem."""
        return -0.669169468

    def _calc_pareto_set(self) -> float:
        """Returns the global minimizer of the problem."""
        return 0.328325636


# instantiate problem and create starting training data
problem = Simple1DProblem()
x0 = [0.19, 0.92]

# instantiate myopic and non-myopic algorithms and then run them on the problem
kwargs = {
    "regression": Idw(),
    "init_points": x0,
    "acquisition_fun_kwargs": {"c1": 1.5078, "c2": 1.4246},
}
algorithms = (
    GO(**kwargs),  # type: ignore[arg-type]
    NonMyopicGO(horizon=2, discount=1.0, **kwargs),  # type: ignore[arg-type]
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
x = np.linspace(*problem.bounds(), 50).reshape(-1, 1)  # type: ignore[call-overload]
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
        a = optimal_acquisition(
            x,
            mdl,
            getattr(algo, "horizon", 1),
            **algo.acquisition_fun_kwargs,
            brute_force=True,
            verbosity=10,
        )
        c = ax.plot(x.flatten(), a, "--", lw=2.5, label="$a(x)$")[0].get_color()
        if i < len(result.history) - 1:
            acq_min = result.history[i + 1].acquisition_min_res.opt.item()
            Xmin, Fmin = acq_min.X[: problem.n_var], acq_min.F[: problem.n_var]
            ax.plot(Xmin, Fmin, "*", markersize=17, color=c)
        else:
            ax.plot(*algo.opt.get("X", "F"), "*", markersize=17, color="k")

        # set axis limits and title
        ax.set_xlim(*problem.bounds())
        ax.set_title(f"iteration {i + 1}, best cost = {ym.min():.4f}", fontsize=9)
        if i == 0:
            ax.set_ylabel(ylbl, fontsize=9)
            if ylbl == "Myopic":
                ax.legend()
    for j in range(i + 1, axs.size):
        axs[j].set_axis_off()
plt.show()
