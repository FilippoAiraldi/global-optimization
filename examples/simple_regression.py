"""
Example of regression with inverse distance weighting (IDW) and radial basis functions
(RBF) on a simple scalar function. This example attempts to reproduce Fig. 1 of [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


import matplotlib.pyplot as plt
import numpy as np

from globopt.core.regression import IDWRegression, RBFRegression

plt.style.use("bmh")


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


# create data points
X = np.array([[-2.61, -1.92, -0.63, 0.38, 2]]).T
y = f(X).flatten()

# create regressors
mdls = [
    IDWRegression(),
    RBFRegression("inversequadratic", 0.5),
    RBFRegression("thinplatespline", 0.01),
]

# fit models to data points - with partial_fit
Xs, ys = np.array_split(X, 3), np.array_split(y, 3)
mdls = [
    mdl.partial_fit(Xs[0], ys[0]).partial_fit(Xs[1], ys[1]).partial_fit(Xs[2], ys[2])
    for mdl in mdls
]
# or with fit
# mdls = [mdl.fit(X, y) for mdl in mdls]

# predict values over all domain via fitted models
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y_hat = [mdl.predict(x) for mdl in mdls]

# plot model predictions
_, ax = plt.subplots(constrained_layout=True, figsize=(7, 3))
ax.plot(x, f(x), label="f(x)")
for mdl, y_hat_ in zip(mdls, y_hat):
    ax.plot(x, y_hat_, label=str(mdl))
ax.plot(X, y, "o")
ax.set_xlabel("x")
ax.set_xlim(-3, 3)
ax.set_ylim(0, 2.5)
ax.legend()
plt.show()
