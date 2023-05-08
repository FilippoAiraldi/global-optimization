"""
Implementation of the non-myopic acquisition function for RBF/IDW-based Global
Optimization, based on the myopic function in [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


import numpy as np

from globopt.core.regression import Array, RegressorType, partial_fit, predict, repeat
from globopt.myopic.acquisition import acquisition as myopic_acquisition


def acquisition(
    x: Array, mdl: RegressorType, c1: float = 1.5078, c2: float = 1.4246
) -> Array:
    """Computes the non-myopic acquisition function for IDW/RBF regression models.

    Parameters
    ----------
    x : array of shape (n_samples, horizon, n_var)
        Array of trajectories for which to compute the acquisition function. `n_samples`
        is the number of units processed in parallel, `horizon` is the horizon length of
        the trajectories, and `n_var` is the number of features/variables per point in
        each trajectory.
    mdl : RegressorType
        Fitted model to use for computing the acquisition function.
    c1 : float, optional
        Weight of the contribution of the variance function, by default `1.5078`.
    c2 : float, optional
        Weight of the contribution of the distance function, by default `1.4246`.

    Returns
    -------
    array of shape (n_samples,)
        The non-myopic acquisition function computed for each trajectory in input `x`.
    """
    n_samples, horizon, _ = x.shape

    # repeat the regressor along the sample dim
    mdl = repeat(mdl, n_samples)

    # loop over the horizon
    a = np.zeros(n_samples, dtype=float)
    for h in range(horizon):
        x_h = x[:, h, np.newaxis, :]
        y_hat = predict(mdl, x_h)
        a += myopic_acquisition(x_h, mdl, y_hat, None, c1, c2)[:, 0, 0]
        mdl = partial_fit(mdl, x_h, y_hat)
    return a
