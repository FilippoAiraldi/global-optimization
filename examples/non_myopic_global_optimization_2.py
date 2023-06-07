"""
Another example comparing the myopic and non-myopic GO algorithms.
"""


import matplotlib.pyplot as plt
import numpy as np
from pymoo.optimize import minimize

from globopt.core.problems import AnotherSimple1DProblem
from globopt.core.regression import Idw, predict
from globopt.myopic.algorithm import GO
from globopt.nonmyopic.acquisition import optimal_acquisition
from globopt.nonmyopic.algorithm import NonMyopicGO

plt.style.use("bmh")


# instantiate problem and create starting training data
problem = AnotherSimple1DProblem()
x0 = [0.19, 0.92]

# instantiate myopic and non-myopic algorithms and then run them on the problem
kwargs = {
    "regression": Idw(),
    "init_points": x0,
    "c1": 1.5078,
    "c2": 1.4246,
}
algorithms = (
    GO(**kwargs),  # type: ignore[arg-type]
    NonMyopicGO(horizon=2, discount=1.0, **kwargs, shrink_horizon=True),
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
    axs[0].set_ylabel(ylbl, fontsize=9)
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
        # if the algorithm has terminated (to do this, we have to access the next
        # iteration of the algorithm)
        if i + 1 < len(result.history):
            next_algo = result.history[i + 1]
            h = getattr(next_algo, "horizon", 1)
            a = optimal_acquisition(x, mdl, h, c1=algo.c1, c2=algo.c2, verbosity=0)
            acq_min = next_algo.acquisition_min_res.opt.item()
            p = acq_min.X[: problem.n_var], acq_min.F[: problem.n_var]

            ax_ = ax.twinx()
            c = ax_.plot(x[:, 0], a, "--", label="$a(x)$", color="C2")[0].get_color()
            ax_.plot(*p, "*", markersize=13, color=c)
            ax_.set_axis_off()
            ylim = ax_.get_ylim()
            ax_.set_ylim(ylim[0] - 0.1, ylim[1] + np.diff(ylim))
        else:
            ax.plot(*algo.opt.get("X", "F"), "*", markersize=17, color="C4")

        # set axis limits and title
        ax.set_xlim(0.001, 0.999)
        ylim = (-0.9, 2.2)
        ax.set_ylim(ylim[0] - np.diff(ylim) * 0.1, ylim[1])
        ax.set_title(f"iteration {i + 1}, best cost = {ym.min():.4f}", fontsize=9)
    for j in range(i + 1, axs.size):
        axs[j].set_axis_off()
plt.show()
