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
    dym: Optional[float] = None,
    c1: float = 1.5078,
    c2: float = 1.4246,
) -> Array:
    """Computes the acquisition function for IDW/RBF regression models.

    Parameters
    ----------
    x : array shape (batch, n_target, n_features)
        Array of points for which to compute the acquisition.
    mdl : Idw or Rbf
        Fitted model to use for computing the acquisition function.
    y_hat : array, optional
        Predictions of the regression model at `x`. If `None`, they are computed based
        on the fitted `mdl`. By default, `None`.
    dym : float or None
        Delta between the maximum and minimum values of `ym`. If `None`, it is computed
        automatically.
    c1 : float
        Weight of the contribution of the variance function.
    c2 : float
        Weight of the contribution of the distance function.

    Returns
    -------
    array
        The variance function acquisition term evaluated at each point.
    """
    Xm = mdl.Xm_
    ym = mdl.ym_
    if y_hat is None:
        y_hat = predict(mdl, x)
    if dym is None:
        dym = ym.max() - ym.min()

    W = idw_weighting(x, Xm, mdl.exp_weighting)
    s = _idw_variance(y_hat, ym, W)
    z = _idw_distance(W)
    return y_hat - c1 * s - c2 * dym * z
