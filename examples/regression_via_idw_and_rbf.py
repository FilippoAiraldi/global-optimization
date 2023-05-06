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

from globopt.core.regression import Idw, Rbf, RegressorType, fit, partial_fit, predict

plt.style.use("bmh")


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


# create data points - X has shape (batch, n_samples, n_var), where the batch dim can be
# used to fit multiple models at once. Here, it is 1
X = np.array([-2.61, -1.92, -0.63, 0.38, 2]).reshape(1, -1, 1)
y = f(X)

# fit regression models to first 3 data points
mdls: list[RegressorType] = [
    Idw(),
    Idw(True),
    Rbf("inversequadratic", 0.5),
    Rbf("thinplatespline", 0.01),
]
mdls = [fit(mdl, X[:, :3], y[:, :3]) for mdl in mdls]

# partially fit regression models to remaining data points
mdls = [partial_fit(mdl, X[:, 3:], y[:, 3:]) for mdl in mdls]

# predict values over all domain via the fitted models
x = np.linspace(-3, 3, 1000).reshape(1, -1, 1)  # again, add batch dim
y_hat = [predict(mdl, x).squeeze() for mdl in mdls]

# plot model predictions
x = x.flatten()
_, ax = plt.subplots(constrained_layout=True, figsize=(7, 3))
ax.plot(x, f(x), label="f(x)", lw=3)
for mdl, y_hat_ in zip(mdls, y_hat):
    ax.plot(x, y_hat_, label=str(mdl))
ax.plot(X.squeeze(), y.squeeze(), "o")
ax.set_xlabel("x")
ax.set_xlim(-3, 3)
ax.set_ylim(0, 2.5)
ax.legend()
plt.show()
