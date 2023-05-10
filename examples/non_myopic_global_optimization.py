"""
Example of application of the GO non-myopic algorithm. The example is taken from Fig. 7
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
from pymoo.util.normalization import NoNormalization

from globopt.benchmarking.optimal_acquisition import optimal_acquisition
from globopt.core.regression import Array, Rbf, predict
from globopt.nonmyopic.algorithm import NonMyopicGO
from globopt.util.normalization import RangeNormalization

plt.style.use("bmh")


class Simple1DProblem(Problem):
    """Simple scalar problem to be minimized."""

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
algorithm = NonMyopicGO(
    regression=Rbf("thinplatespline", 0.01),
    init_points=x0,
    horizon=2,
    discount=1.0,
    acquisition_min_kwargs={"verbose": True},
    acquisition_fun_kwargs={"c1": 1, "c2": 0.5},
)
res = minimize(
    problem,
    algorithm,
    termination=("n_iter", 4),
    verbose=True,
    seed=1,
    save_history=True,
)

# plot the results
x = np.linspace(*problem.bounds(), 100).reshape(-1, 1)  # type: ignore[call-overload]
y = problem.evaluate(x)
_, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(10, 2))
axs = axs.flatten()
for i, (ax, algo) in enumerate(zip(axs, res.history)):
    # plot true function and current sampled points
    Xm = algo.pop.get("X").reshape(-1, 1)
    ym = algo.pop.get("F").reshape(-1)
    c = ax.plot(x, y, label="$f(x)$")[0].get_color()
    ax.plot(Xm, ym, "o", color=c, markersize=8)

    # plot current regression model's prediction
    mdl = algo.regression
    y_hat = predict(mdl, x[np.newaxis])[0]
    ax.plot(x, y_hat, label=r"$\hat{f}(x)$")

    # plot the optimal acquisition function and its minimum
    a = optimal_acquisition(
        x,
        mdl,
        algo.horizon,
        **algo.acquisition_fun_kwargs,
        brute_force=True,
        verbosity=10,
    )
    ax.plot(x.flatten(), a, label="$a(x)$")
    if i < len(axs) - 1:
        acq_min = res.history[i + 1].acquisition_min_res.opt.item()
        Xmin = acq_min.X[: problem.n_var]
        ax.plot(Xmin, problem.evaluate(Xmin), "*", markersize=13, color="k")

    # set axis limits and title
    ax.set_xlim(*problem.bounds())
    ax.set_ylim(0, 2.5)
    ax.set_title(f"iter = {i + 1}, best cost = {ym.min():.4f}", fontsize=9)
    if i == 0:
        ax.legend()
for j in range(i + 1, len(axs)):
    axs[j].set_axis_off()
plt.show()
