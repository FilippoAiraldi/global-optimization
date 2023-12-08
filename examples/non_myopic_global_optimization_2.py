"""
Another example comparing the myopic and non-myopic GO algorithms.
"""


from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from vpso.typing import Array1d

from globopt.core.problems import AnotherSimple1dProblem
from globopt.core.regression import Idw, predict
from globopt.myopic.acquisition import acquisition as myopic_acquisition
from globopt.myopic.algorithm import go
from globopt.nonmyopic.acquisition import acquisition as nonmyopic_acquisition
from globopt.nonmyopic.algorithm import nmgo

plt.style.use("bmh")


# instantiate problem and create starting training data
f = AnotherSimple1dProblem.f
lb = AnotherSimple1dProblem.lb
ub = AnotherSimple1dProblem.ub
x0 = [0.08, 0.17, 0.92]
c1 = 1.0
c2 = 0.5

# helper quantities and method
history: list[
    tuple[Array1d, Array1d, float, float, Array1d, Array1d, Array1d, Array1d]
] = []
x = np.linspace(lb, ub, 300)


def save_history(algorithm: Literal["go", "nmgo"], locals: dict[str, Any]) -> None:
    iteration = locals.get("iteration", 0)
    print(f"{algorithm} - iteration {iteration}")
    if iteration > 0:
        x_ = x.reshape(1, -1, 1)
        mdl = locals["mdl"]
        y_hat = predict(mdl, x_)
        if algorithm == "go":
            acq = myopic_acquisition(x_, mdl, c1, c2, y_hat, None).squeeze()
        else:
            acq = nonmyopic_acquisition(
                x_.transpose(1, 0, 2),
                mdl,
                locals["horizon"],
                locals["discount"],
                lb,
                ub,
                c1,
                c2,
                locals["mc_iters"],
                locals["quasi_mc"],
                locals["common_random_numbers"],
                locals["antithetic_variates"],
                locals["pso_kwargs"],
                False,
                0,
                {"n_jobs": -1, "backend": "loky"},
            )
        history.append(
            (
                locals["x_new"],
                locals["y_new"],
                locals["y_best"],
                acq,
                locals["acq_opt"],
                y_hat.squeeze(),
                mdl.Xm_.squeeze(),
                mdl.ym_.squeeze(),
            )
        )


# run the myopic optimization
ITERS = 4
pso_kwargs = {
    "swarmsize": 10,
    "xtol": -1,
    "ftol": 1e-4,
    "maxiter": 300,
    "patience": 1,
}
x_best, y_best = go(
    func=f,
    lb=lb,
    ub=ub,
    mdl=Idw(),
    init_points=x0,
    c1=c1,
    c2=c2,
    maxiter=ITERS,
    seed=1909,
    callback=save_history,
    pso_kwargs=pso_kwargs,
)
myopic_history = history.copy()
history.clear()

# run the non-myopic optimization
x_best, y_best = nmgo(
    func=f,
    lb=lb,
    ub=ub,
    mdl=Idw(),
    horizon=2,
    discount=1.0,
    init_points=x0,
    c1=c1,
    c2=c2,
    mc_iters=2**10,
    parallel={"n_jobs": -1, "backend": "loky", "verbose": 1},
    maxiter=ITERS,
    seed=1909,
    callback=save_history,
    pso_kwargs=pso_kwargs,
)
nonmyopic_history = history.copy()
history.clear()


# plot the results
x = x.flatten()
y = f(x)
n_cols = min(ITERS, 4)
n_rows = int(np.ceil(ITERS / n_cols)) * 2
_, all_axs = plt.subplots(
    n_rows,
    n_cols,
    constrained_layout=True,
    figsize=(2.5 * n_cols, 2 * n_rows),
    sharex=True,
    sharey=True,
)
for history, ylbl, axs in zip(
    (myopic_history, nonmyopic_history), ("Myopic", "Non-myopic"), np.split(all_axs, 2)
):
    axs = axs.flatten()
    axs[0].set_ylabel(ylbl, fontsize=9)
    for i, (ax, history_item) in enumerate(zip(axs, history)):
        x_new, y_new, y_best, acq, acq_opt, y_hat, Xm, ym = history_item

        # plot true function, current sampled points, and regression prediction
        ax.plot(x, y, label="$f(x)$", color="C0")
        ax.plot(Xm, ym, "o", color="C0", markersize=8)
        ax.plot(x, y_hat, label=r"$\hat{f}(x)$", color="C1")

        # plot acquisition function and its minimum
        ax_ = ax.twinx()
        ax_.plot(x.reshape(-1), acq, "--", lw=2.5, label="$a(x)$", color="C2")
        ax_.plot(x_new, acq_opt, "*", markersize=13, color="C2")
        ax_.set_axis_off()
        ylim = ax_.get_ylim()
        ax_.set_ylim(ylim[0] - 0.1, ylim[1] + np.diff(ylim) * 0.7)

        # plot next point
        ax.plot(x_new, y_new, "o", markersize=8, color="C4")

        # set axis limits and title
        ax.set_xlim(0.001, 0.999)
        ylim = (-0.9, 2.2)
        ax.set_ylim(ylim[0] - np.diff(ylim) * 0.1, ylim[1])
        ax.set_title(f"iter = {i + 1}, best cost = {y_best:.4f}", fontsize=9)
    for j in range(i + 1, axs.size):
        axs[j].set_axis_off()
plt.show()
