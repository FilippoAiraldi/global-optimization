"""
Implementation of the acquisition function for RBF/IDW-based Global Optimization
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Optional

import numpy as np

from globopt.core.regression import Array, RegressorType, idw_weighting, predict


def _idw_variance(y_hat: Array, ym: Array, W: Array) -> Array:
    """Computes the variance function acquisition term for IDW/RBF regression models.

    Parameters
    ----------
    y_hat : array
        Array of regressed predictions for which to compute the variance.
    ym : array
        Dataset of `y` used to fit the regression model.
    W : array
        Weights computed via `idw_weighting`.

    Returns
    -------
    array
        The variance function acquisition term evaluated at each point.
    """
    V = W / W.sum(2, keepdims=True)
    sqdiff = np.square(ym - y_hat.transpose(0, 2, 1))
    out = np.diagonal(V @ sqdiff, axis1=1, axis2=2)[..., None]  # fast for small arrays
    # out = np.empty_like(y_hat)
    # for i in range(out.shape[1]):
    #     out[:, i] = np.diag(V[:, i] @ sqdiff[:, :, i].T).reshape(-1, 1)
    return np.sqrt(out)


def _idw_distance(W: Array) -> Array:
    """Computes the distance function acquisition term for IDW/RBF regression models.

    Parameters
    ----------
    W : array
        Weights computed via `idw_weighting`.

    Returns
    -------
    array
        The distance function acquisition term evaluated at each point.
    """
    return (2 / np.pi) * np.arctan(1 / W.sum(2, keepdims=True))


def acquisition(
    x: Array,
    mdl: RegressorType,
    y_hat: Optional[Array] = None,
    dym: Optional[Array] = None,
    c1: float = 1.5078,
    c2: float = 1.4246,
) -> Array:
    """Computes the myopic acquisition function for IDW/RBF regression models.

    Parameters
    ----------
    x : array of shape (batch, n_samples, n_var)
        Array of points for which to compute the acquisition. `batch` is the number of
        parallel regressors used in the computations, `n_samples` is the number of
        target points for which to compute the acquisition, and `n_var` is the number of
        features/variables of each point.
    mdl : Idw or Rbf
        Fitted model (or batch of fitted models) to use for computing the acquisition
        function.
    y_hat : array of shape (batch, n_samples, 1), optional
        Predictions of the regression model at `x`. If `None`, they are computed based
        on the fitted `mdl`. If pre-computed, can be provided to speed up computations;
        otherwise is computed on-the-fly automatically. By default, `None`.
    dym : array of shape (batch, 1, 1), optional
        Delta between the maximum and minimum values of `ym`. If pre-computed, can be
        provided to speed up computations; otherwise is computed on-the-fly
        automatically. If `None`, it is computed automatically. By default, `None`.
    c1 : float, optional
        Weight of the contribution of the variance function, by default `1.5078`.
    c2 : float, optional
        Weight of the contribution of the distance function, by default `1.4246`.

    Returns
    -------
    array of shape (batch, n_samples, 1)
        The myopic acquisition function evaluated at each point.
    """
    Xm = mdl.Xm_
    ym = mdl.ym_
    if y_hat is None:
        y_hat = predict(mdl, x)
    if dym is None:
        dym = ym.ptp((1, 2), keepdims=True)

    W = idw_weighting(x, Xm, mdl.exp_weighting)
    s = _idw_variance(y_hat, ym, W)
    z = _idw_distance(W)
    return y_hat - c1 * s - c2 * dym * z
