"""
Example of application of the GO myopic algorithm. This example attempts to reproduce
Fig. 7 of [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from vpso.typing import Array1d

from globopt.core.problems import simple1dproblem
from globopt.core.regression import Idw, Rbf, predict
from globopt.myopic.algorithm2 import acquisition, go

plt.style.use("bmh")


# instantiate problem and create starting training data
f = simple1dproblem.f
lb = simple1dproblem.lb
ub = simple1dproblem.ub
x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
c1 = 1
c2 = 0.5

# helper quantities and method
history: list[tuple[Array1d, float, float, Array1d, Array1d, Array1d, Array1d]] = []
x = np.linspace(lb, ub, 300)


def save_history(
    iter: int,
    x_best: Array1d,
    y_best: float,
    x_new: Array1d,
    y_new: float,
    acq_opt: float,
    mdl: Union[Idw, Rbf],
    mdl_new: Union[Idw, Rbf],
) -> None:
    if iter > 0:
        x_ = x.reshape(1, -1, 1)
        y_hat = predict(mdl, x_)
        acq = acquisition(x_, mdl, y_hat, None, c1, c2)[0, :, 0]
        Xm, ym = mdl.Xm_.squeeze(), mdl.ym_.squeeze()
        history.append((x_new, y_best, acq, acq_opt, y_hat.squeeze(), Xm, ym))


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
    x_new, y_best, acq, acq_opt, y_hat, Xm, ym = history_item

    # plot true function, current sampled points, and regression prediction
    c = ax.plot(x, y, label="$f(x)$")[0].get_color()
    ax.plot(Xm, ym, "o", color=c, markersize=8)
    ax.plot(x, y_hat, label=r"$\hat{f}(x)$")

    # plot acquisition function and its minimum
    ax_ = ax.twinx()
    line = ax_.plot(x.reshape(-1), acq, "--", lw=2.5, label="$a(x)$", color="C2")
    ax_.plot(x_new, acq_opt, "*", markersize=13, color=line[0].get_color())
    ax_.set_axis_off()
    ylim = ax_.get_ylim()
    ax_.set_ylim(ylim[0] - 0.1, ylim[1] + np.diff(ylim) * 0.7)

    # set axis limits and title
    ax.set_xlim(lb, ub)
    ylim = (0.1, 2.4)
    ax.set_ylim(ylim[0] - np.diff(ylim) * 0.1, ylim[1])
    ax.set_title(f"iter = {i + 1}, best cost = {y_best:.4f}", fontsize=9)
for j in range(i + 1, len(axs)):
    axs[j].set_axis_off()
plt.show()
