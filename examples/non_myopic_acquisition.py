"""Example application of the non-myopic acquisition function."""


import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from globopt.benchmarking.optimal_acquisition import optimal_acquisition
from globopt.core.regression import Array, Rbf, RegressorType, fit, predict
from globopt.myopic.acquisition import acquisition as myopic_acquisition
from globopt.nonmyopic.acquisition import acquisition as nonmyopic_acquisition

plt.style.use("bmh")

# define the function and its domain
xl, xu = -3, +3


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


# create functions for computing and optimizing the myopic and non-myopic acquisitions


def compute_myopic_acquisition(
    x: Array, mdl: RegressorType, c1: float, c2: float
) -> tuple[Array, tuple[float, float]]:
    X, y = mdl.Xm_, mdl.ym_
    dym = y.max() - y.min()
    n_var = X.shape[-1]

    def obj(x_: Array) -> Array:
        return myopic_acquisition(x_[np.newaxis], mdl, None, dym, c1, c2)[0]

    problem = FunctionalProblem(n_var=n_var, objs=obj, xl=xl, xu=xu, elementwise=False)
    res = minimize(problem, PSO(25), verbose=True, seed=1).opt[0]
    a = obj(x[0])
    return a, (res.X, res.F)


def compute_nonmyopic_acquisition(
    x: Array, mdl: RegressorType, h: int, c1: float, c2: float, discount: float
) -> tuple[Array, tuple[float, float]]:
    n_var = mdl.Xm_.shape[-1]

    def obj(x_: Array) -> Array:
        # transform x_ from (n_samples, n_var * h) to (n_samples, h, n_var)
        x_ = x_.reshape(-1, h, n_var)
        return nonmyopic_acquisition(x_, mdl, c1, c2, discount)

    problem = FunctionalProblem(
        n_var=n_var * h, objs=obj, xl=xl, xu=xu, elementwise=False
    )
    res = minimize(problem, PSO(25 * h), verbose=True, seed=1).opt[0]
    a = optimal_acquisition(x[0], mdl, h, c1=c1, c2=c2, brute_force=True, verbosity=10)
    return a, (res.X[:n_var], res.F)


# create data points - X has shape (batch, n_samples, n_var), where the batch dim can be
# used to fit multiple models at once. Here, it is 1
X = np.array([-2.62, -1.99, 0.14, 1.01, 2.62]).reshape(1, -1, 1)
y = f(X)
mdl = fit(Rbf("thinplatespline", 0.01), X, y)

# compute myopic acquisition function
c1 = 1.0
c2 = 0.5
x = np.linspace(xl, xu, 100).reshape(1, -1, 1)  # add batch dim
myopic_results = compute_myopic_acquisition(x, mdl, c1, c2)

# compute non-myopic acquisition function
horizon = 2
discount = 1.0
nonmyopic_results = compute_nonmyopic_acquisition(x, mdl, horizon, c1, c2, discount)

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
