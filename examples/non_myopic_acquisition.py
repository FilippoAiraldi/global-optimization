"""Example application of the non-myopic acquisition function."""


import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from globopt.core.problems import Simple1DProblem
from globopt.core.regression import Array, Rbf, RegressorType, fit, predict
from globopt.myopic.acquisition import acquisition as myopic_acquisition
from globopt.nonmyopic.acquisition import acquisition as nonmyopic_acquisition

plt.style.use("bmh")

# define the function and its domain
xl, xu = -3, +3
f = Simple1DProblem.f


# create functions for computing and optimizing the myopic and non-myopic acquisitions


def compute_myopic_acquisition(
    x: Array, mdl: RegressorType, c1: float, c2: float
) -> tuple[Array, tuple[float, float]]:
    # compute the myopic acquisition function for x
    X, y = mdl.Xm_, mdl.ym_
    dym = y.ptp()
    a = myopic_acquisition(x, mdl, None, dym, c1, c2)

    # find the minimium of the myopic acquisition function
    n_var = X.shape[-1]
    problem = FunctionalProblem(
        n_var,
        lambda x_: myopic_acquisition(x_, mdl, None, dym, c1, c2),
        xl=xl,
        xu=xu,
        elementwise=False,
    )
    res = minimize(problem, PSO(25), verbose=True, seed=1).opt[0]
    return a, (res.X, res.F)


def compute_nonmyopic_acquisition(
    x: Array, mdl: RegressorType, h: int, c1: float, c2: float, discount: float
) -> tuple[Array, tuple[float, float]]:
    # compute the non-myopic acquisition function for x
    with Parallel(n_jobs=-1, batch_size=8, verbose=10) as parallel:
        a = nonmyopic_acquisition(
            x, mdl, h, discount, c1, c2, xl=xl, xu=xu, parallel=parallel
        )

        # find the minimium of the myopic acquisition function
        parallel.verbose = 0
        n_var = X.shape[-1]
        problem = FunctionalProblem(
            n_var,
            lambda x_: nonmyopic_acquisition(
                x_, mdl, h, discount, c1, c2, xl=xl, xu=xu, parallel=parallel
            ),
            xl=xl,
            xu=xu,
            elementwise=False,
        )
        res = minimize(problem, PSO(25), verbose=True, seed=1).opt[0]
        return a, (res.X, res.F)


# create data points - X has shape (n_samples, n_var)
X = np.array([-2.62, -1.99, 0.14, 1.01, 2.62]).reshape(-1, 1)
y = f(X).reshape(-1)
mdl = fit(Rbf("thinplatespline", 0.01), X, y)

# compute myopic acquisition function
c1 = 1.0
c2 = 0.5
x = np.linspace(xl, xu, 300).reshape(-1, 1)
myopic_results = compute_myopic_acquisition(x, mdl, c1, c2)

# compute non-myopic acquisition function
horizon = 3
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
