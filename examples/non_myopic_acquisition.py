"""Example application of the non-myopic acquisition function."""


import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from globopt.core.regression import Array, Idw, RegressorType, fit, predict
from globopt.myopic.acquisition import acquisition as myopic_acquisition
from globopt.nonmyopic.acquisition import acquisition as nonmyopic_acquisition
from globopt.util.optimal_acquisition import optimal_acquisition

plt.style.use("bmh")


def f(x):
    return np.sin(20 * x) + 20 * np.square(x - 0.3)


def compute_myopic_acquisition(
    mdl: RegressorType, x: Array
) -> tuple[Array, tuple[float, float]]:
    # compute minimizer of acquisition function
    X, y = mdl.Xm_, mdl.ym_
    dym = y.max() - y.min()
    n_var = X.shape[-1]

    # elementwise=False enables vectorized evaluation of acquisition function
    def obj(x_: Array) -> Array:
        return myopic_acquisition(x_[np.newaxis], mdl, dym=dym, c1=1, c2=0.5)[0]

    problem = FunctionalProblem(n_var=n_var, objs=obj, xl=0, xu=1, elementwise=False)
    res = minimize(problem, PSO(), verbose=True, seed=1).opt[0]
    return obj(x[0]), (res.X, res.F)


def compute_nonmyopic_acquisition(
    mdl: RegressorType, x: Array, h: int
) -> tuple[Array, tuple[float, float]]:
    # compute minimizer of non-myopic acquisition function
    n_var = mdl.Xm_.shape[-1]

    def obj(x_: Array) -> Array:
        # transform x_ from (n_samples, n_var * h) to (n_samples, h, n_var)
        x_ = x_.reshape(-1, h, n_var)
        return nonmyopic_acquisition(x_, mdl, c1=1, c2=0.5)

    problem = FunctionalProblem(
        n_var=n_var * h, objs=obj, xl=0, xu=1, elementwise=False
    )
    res = minimize(problem, PSO(25 * h), verbose=True, seed=1).opt[0]

    return optimal_acquisition(x[0], mdl, h, c1=1, c2=0.5), (res.X[:n_var], res.F)


# create data points - X has shape (batch, n_samples, n_var), where the batch dim can be
# used to fit multiple models at once. Here, it is 1
X = np.array([0.1141, 0.3706, 0.5602, 0.8178, 0.9441]).reshape(1, -1, 1)
y = f(X)

# compute myopic acquisition function
mdl = fit(Idw(), X, y)
x = np.linspace(0, 1, 60).reshape(1, -1, 1)
myopic_results = compute_myopic_acquisition(mdl, x)

# compute non-myopic acquisition function
horizon = 3
nonmyopic_results = compute_nonmyopic_acquisition(mdl, x, horizon)

# plot function and its estimate
_, ax = plt.subplots(constrained_layout=True, figsize=(9, 4))
y_hat = predict(mdl, x)
line = ax.plot(x.flatten(), f(x).flatten(), label=r"$f(x)$")[0]
ax.plot(X.flatten(), y.flatten(), "o", label=None, color=line.get_color(), markersize=7)
ax.plot(x.flatten(), y_hat.flatten(), label=r"$\hat{f}(x)$")

# plot acquisition functions
for (a, a_min), lbl in zip(
    (myopic_results, nonmyopic_results), ("Myopic", "Non-myopic")
):
    line = ax.plot(x.flatten(), a.flatten(), "--", label=f"{lbl} $a(x)$")[0]
    ax.plot(*a_min, "*", label=None, color=line.get_color(), markersize=12)

# make plot pretty
ax.set_xlim(0, 1)
ax.set_xlabel("x")
ax.legend()
plt.show()
