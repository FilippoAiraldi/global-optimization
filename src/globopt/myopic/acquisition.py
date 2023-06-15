"""
Implementation of the acquisition function for RBF/IDW-based Global Optimization
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Optional

import numba as nb
import numpy as np
from vpso.jit import _float, jit
from vpso.typing import Array3d

from globopt.core.regression import RegressorType, _idw_weighting, predict


@jit(_float[:, :, :](_float[:, :, :], _float[:, :, :]), parallel=True)
def _idw_variance_inner_loop(V, sqdiff):
    """Parallelized inner loop of the variance function."""
    B, n, _ = V.shape
    out = np.empty((B, n, 1))
    for i in nb.prange(n):
        out[:, i, 0] = np.diag(V[:, i] @ sqdiff[:, :, i].T)
    return out


@jit(_float[:, :, :](_float[:, :, :], _float[:, :, :], _float[:, :, :]))
def _idw_variance(y_hat: Array3d, ym: Array3d, W: Array3d) -> Array3d:
    """Computes the variance function acquisition term for IDW/RBF regression models."""
    V = W / W.sum(2)[:, :, np.newaxis]
    sqdiff = np.square(ym - y_hat.transpose(0, 2, 1))
    return np.sqrt(_idw_variance_inner_loop(V, sqdiff))


@jit(_float[:, :, :](_float[:, :, :]))
def _idw_distance(W: Array3d) -> Array3d:
    """Computes the distance function acquisition term for IDW/RBF regression models."""
    return ((2 / np.pi) * np.arctan(1 / W.sum(2)))[:, :, np.newaxis]


@jit(
    _float[:, :, :](
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float,
        _float,
        nb.boolean,
    )
)
def _compute_acquisition(
    Xm: Array3d,
    x: Array3d,
    ym: Array3d,
    y_hat: Array3d,
    dym: Array3d,
    c1: float,
    c2: float,
    exp_weighting: bool,
) -> Array3d:
    """Runs the computations of the myopic acquisition function for IDW/RBF models."""
    W = _idw_weighting(x, Xm, exp_weighting)
    s = _idw_variance(y_hat, ym, W)
    z = _idw_distance(W)
    return y_hat - c1 * s - c2 * dym * z


def acquisition(
    x: Array3d,
    mdl: RegressorType,
    y_hat: Optional[Array3d] = None,
    dym: Optional[Array3d] = None,
    c1: float = 1.5078,
    c2: float = 1.4246,
) -> Array3d:
    """Computes the myopic acquisition function for IDW/RBF regression models.

    Parameters
    ----------
    x : array of shape (batch, n_samples, n_var)
        Array of points for which to compute the acquisition. `batch` is the dimension
        of the batched regressor model (i.e., multiple regressors batched together),
        `n_samples` is the number of target points for which to compute the acquisition,
        and `n_var` is the number
        of features/variables of each point.
    mdl : Idw or Rbf
        Fitted model to use for computing the acquisition function.
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
    return _compute_acquisition(Xm, x, ym, y_hat, dym, c1, c2, mdl.exp_weighting)
