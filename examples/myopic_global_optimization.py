"""
Example of application of the GO myopic algorithm. This example attempts to reproduce
Fig. 7 of [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from vpso.typing import Array1d

from globopt.core.problems import Simple1dProblem
from globopt.core.regression import Rbf, predict
from globopt.myopic.algorithm2 import acquisition, go

plt.style.use("bmh")


# instantiate problem and create starting training data
f = Simple1dProblem.f
lb = Simple1dProblem.lb
ub = Simple1dProblem.ub
x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
c1 = 1
c2 = 0.5

# helper quantities and method
history: list[
    tuple[Array1d, Array1d, float, float, Array1d, Array1d, Array1d, Array1d]
] = []
x = np.linspace(lb, ub, 300)


def save_history(_: Literal["go", "nmgo"], locals: dict[str, Any]) -> None:
    if locals.get("iteration", 0) > 0:
        x_ = x.reshape(1, -1, 1)
        mdl = locals["mdl"]
        y_hat = predict(mdl, x_)
        history.append(
            (
                locals["x_new"],
                locals["y_new"],
                locals["y_best"],
                acquisition(x_, mdl, y_hat, None, c1, c2).squeeze(),
                locals["acq_opt"],
                y_hat.squeeze(),
                mdl.Xm_.squeeze(),
                mdl.ym_.squeeze(),
            )
        )


# run the optimization
x_best, y_best = go(
    func=f,
    lb=lb,
    ub=ub,
    mdl=Rbf("thinplatespline", 0.01),
    init_points=x0,
    c1=c1,
    c2=c2,
    maxiter=6,
    seed=1909,
    callback=save_history,
)


# plot the results
x = x.flatten()
y = f(x)
n_cols = 4
n_rows = int(np.ceil((len(history) + 1) / n_cols))
_, axs = plt.subplots(
    n_rows,
    n_cols,
    constrained_layout=True,
    figsize=(2.5 * n_cols, 2 * n_rows),
    sharex=True,
    sharey=True,
)
axs = axs.flatten()
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
    ax.set_xlim(lb, ub)
    ylim = (0.1, 2.4)
    ax.set_ylim(ylim[0] - np.diff(ylim) * 0.1, ylim[1])
    ax.set_title(f"iter = {i + 1}, best cost = {y_best:.4f}", fontsize=9)
for j in range(i + 1, len(axs)):
    axs[j].set_axis_off()
plt.show()
