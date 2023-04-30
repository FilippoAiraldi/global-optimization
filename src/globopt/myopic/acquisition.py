from typing import Optional

import numpy as np
import numpy.typing as npt
from numba import njit

from globopt.core.regression import idw_weighting

TWO_DIV_PI = 2 / np.pi


@njit
def idw_variance(
    y_hat: npt.NDArray[np.floating],
    ym: npt.NDArray[np.floating],
    W: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
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
    V = W / W.sum(axis=0)
    sqdiff = np.square(ym.reshape(-1, 1) - y_hat.reshape(1, -1))  # np.subtract.outer
    out = np.empty(y_hat.shape)
    for i in range(len(out)):
        out[i] = V[:, i].T @ sqdiff[:, i]
    return np.sqrt(out)
    # return np.sqrt(np.diag(V.T @ sqdiff))  # faster for small arrays


# no need to jit this function, it's already fast enough
def idw_distance(W: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
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
    return TWO_DIV_PI * np.arctan(1 / W.sum(axis=0))


# cannot jit this function due to idw_weighting
def acquisition(
    x: npt.NDArray[np.floating],
    y_hat: npt.NDArray[np.floating],
    Xm: npt.NDArray[np.floating],
    ym: npt.NDArray[np.floating],
    dym: Optional[float],
    c1: float,
    c2: float,
    exp_weighting: bool = False,
) -> npt.NDArray[np.floating]:
    """Computes the acquisition function for IDW/RBF regression models.

    Parameters
    ----------
    x : array
        Array of points for which to compute the acquisition.
    y_hat : array
        Array of regressed predictions for which to compute the variance.
    Xm : array
        Dataset of `X` used to fit the regression model.
    ym : array
        Dataset of `y` used to fit the regression model.
    dym : float or None
        Delta between the maximum and minimum values of `ym`. If `None`, it is computed
        automatically.
    c1 : float
        Weight of the contribution of the variance function.
    c2 : float
        Weight of the contribution of the distance function.
    exp_weighting : bool, optional
        Whether the weighting function should decay exponentially, by default `False`.
    Returns
    -------
    array
        The variance function acquisition term evaluated at each point.
    """
    # compute variance and distance functions
    W = idw_weighting(x, Xm, exp_weighting)
    s = idw_variance(y_hat, ym, W)
    z = idw_distance(W)

    # compute acquisition function
    if dym is None:
        dym = ym.max() - ym.min()
    return y_hat - c1 * s - c2 * dym * z
