"""Example application of the non-myopic acquisition function."""


from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from vpso import vpso
from vpso.typing import Array3d

from globopt.core.problems import Simple1dProblem
from globopt.core.regression import Rbf, RegressorType, fit, predict
from globopt.myopic.acquisition import acquisition as myopic_acquisition
from globopt.nonmyopic.acquisition import acquisition as nonmyopic_acquisition
from globopt.nonmyopic.helpers import mpc_acquisition_by_brute_force

plt.style.use("bmh")

# define the function and its domain
lb, ub = -3, +3
f = Simple1dProblem.f


# create functions for computing and optimizing the myopic and non-myopic acquisitions


def compute_myopic_acquisition(
    x: Array3d, mdl: RegressorType, c1: float, c2: float
) -> tuple[Array3d, tuple[float, float]]:
    # compute the myopic acquisition function for x
    dym = mdl.ym_.ptp(1, keepdims=True)
    a = myopic_acquisition(x, mdl, c1, c2, None, dym)

    # find the minimium of the myopic acquisition function
    x_opt, y_opt, _ = vpso(
        func=lambda x: myopic_acquisition(x, mdl, c1, c2)[..., 0],
        lb=np.array([[lb]]),
        ub=np.array([[ub]]),
        seed=1909,
    )
    return a, (x_opt.item(), y_opt.item())


def compute_nonmyopic_acquisition(
    x: Array3d,
    mdl: RegressorType,
    h: int,
    c1: float,
    c2: float,
    discount: float,
    type: Literal["rollout", "mpc"],
) -> tuple[Array3d, tuple[float, float]]:
    # compute the non-myopic acquisition function for x
    if type == "mpc":
        a = mpc_acquisition_by_brute_force(x[0], mdl, h, c1, c2, discount, 100)
    else:
        a = nonmyopic_acquisition(
            x.transpose(1, 0, 2),
            mdl,
            h,
            discount,
            c1,
            c2,
            "rollout",
            lb=np.asarray([lb]),
            ub=np.asarray([ub]),
        )

    # TODO: find the minimium of the non-myopic acquisition function
    return a, (1, 1)


# create data points - X has shape (batch, n_samples, dim), with batch=1
X = np.array([-2.62, -1.99, 0.14, 1.01, 2.62]).reshape(1, -1, 1)
y = f(X)
mdl = fit(Rbf("thinplatespline", 0.01), X, y)

# compute myopic acquisition function
c1 = 1.0
c2 = 0.5
x = np.linspace(lb, ub, 100).reshape(1, -1, 1)
myopic_results = compute_myopic_acquisition(x, mdl, c1, c2)

# compute non-myopic acquisition function
horizon = 3
discount = 1.0
nonmyopic_results = compute_nonmyopic_acquisition(
    x, mdl, horizon, c1, c2, discount, "mpc"
)
# plot function and its estimate
_, ax = plt.subplots(constrained_layout=True, figsize=(5, 3))
y_hat = predict(mdl, x)
line = ax.plot(x.flatten(), f(x).flatten(), label=r"$f(x)$")[0]
ax.plot(X.flatten(), y.flatten(), "o", label=None, color=line.get_color(), markersize=9)
ax.plot(x.flatten(), y_hat.flatten(), label=r"$\hat{f}(x)$")

# plot acquisition functions
for (a, a_min), lbl in zip(
    (myopic_results, nonmyopic_results), ("Myopic", "Non-myopic")
):
    line = ax.plot(x.flatten(), a.flatten(), "--", lw=2.5, label=f"{lbl} $a(x)$")[0]
    ax.plot(*a_min, "*", label=None, color=line.get_color(), markersize=17)

# make plot pretty
ax.set_xlim(x.min(), x.max())
ax.set_xlabel("x")
ax.legend()
plt.show()
