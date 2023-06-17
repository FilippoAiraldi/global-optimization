"""Example application of the non-myopic acquisition function."""


import matplotlib.pyplot as plt
import numpy as np

from globopt.core.problems import Simple1dProblem
from globopt.core.regression import Rbf, fit, predict
from globopt.myopic.acquisition import acquisition as myopic_acquisition
from globopt.nonmyopic.acquisition import (
    deterministic_acquisition as nonmyopic_deterministic_acquisition,
)

plt.style.use("bmh")

# define the function and its domain
lb, ub = np.array([-3.0]), np.array([+3.0])
f = Simple1dProblem.f

# create data points - X has shape (batch, n_samples, dim), with batch=1
X = np.array([-2.62, -1.99, 0.14, 1.01, 2.62]).reshape(1, -1, 1)
y = f(X)
mdl = fit(Rbf("thinplatespline", 0.01), X, y)

# compute myopic acquisition function
c1 = 1.0
c2 = 0.5
x = np.linspace(lb, ub, 100).reshape(-1, 1, 1)
myopic_a = myopic_acquisition(x.transpose(1, 0, 2), mdl, c1, c2).squeeze()

# compute deterministic non-myopic acquisition function
horizon = 3
discount = 1.0
nonmyopic_deterministic_a = nonmyopic_deterministic_acquisition(
    x, mdl, horizon, discount, c1, c2, "rollout", lb=lb, ub=ub
).squeeze()


# TODO: compute MC non-myopic acquisition function


# plot function, its estimate and the acquisition functions
_, ax = plt.subplots(constrained_layout=True, figsize=(5, 3))
y_hat = predict(mdl, x.transpose(1, 0, 2))
x = x.squeeze()
ax.plot(x, f(x), label=r"$f(x)$", color="C0")
ax.plot(X.squeeze(), y.squeeze(), "o", label=None, color="C0", markersize=9)
ax.plot(x, y_hat.flatten(), label=r"$\hat{f}(x)$", color="C1")
ax_ = ax.twinx()
for i, (a, lbl) in enumerate(
    zip((myopic_a, nonmyopic_deterministic_a), ("Myopic", "Non-myopic")), start=2
):
    ax_.plot(x, a, "--", lw=2.5, label=f"{lbl} $a(x)$", color=f"C{i}")
    k = np.argmin(a)
    ax_.plot(x[k], a[k], "*", label=None, color=f"C{i}", markersize=17)

# make plot pretty
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-0.2, ax.get_ylim()[1])
ax.set_xlabel("x")
ax.legend()
ax_.set_axis_off()
ax_.set_xlim(x.min(), x.max())
ylim = ax_.get_ylim()
ax_.set_ylim(ylim[0] - 0.1, ylim[1] + np.diff(ylim) * 0.7)
plt.show()
