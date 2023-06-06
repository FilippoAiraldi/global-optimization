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

from globopt.core.problems import Simple1DProblem
from globopt.core.regression import Idw, Rbf, RegressorType, fit, partial_fit, predict

plt.style.use("bmh")


f = Simple1DProblem.f


# create data points - X has shape (n_samples, n_var)
X = np.array([-2.61, -1.92, -0.63, 0.38, 2]).reshape(-1, 1)
y = f(X).reshape(-1)

# fit regression models to first 3 data points
mdls: list[RegressorType] = [
    Idw(),
    Idw(True),
    Rbf("inversequadratic", 0.5),
    Rbf("thinplatespline", 0.01),
]
mdls = [fit(mdl, X[:3], y[:3]) for mdl in mdls]

# partially fit regression models to remaining data points
mdls = [partial_fit(mdl, X[3:], y[3:]) for mdl in mdls]

# predict values over all domain via the fitted models
x = np.linspace(-3, 3, 1000).reshape(-1, 1)
y_hat = [predict(mdl, x).squeeze() for mdl in mdls]

# plot model predictions
x = x.flatten()
_, ax = plt.subplots(constrained_layout=True, figsize=(7, 3))
ax.plot(X.squeeze(), y.squeeze(), "o", markersize=9, color="C0")
ax.plot(x, f(x), label="f(x)", lw=3)
for mdl, y_hat_ in zip(mdls, y_hat):
    ax.plot(x, y_hat_, label=str(mdl))
ax.set_xlabel("x")
ax.set_xlim(-3, 3)
ax.set_ylim(0, 2.5)
ax.legend()
plt.show()
