"""
Implementation of Radial Basis Function and Inverse Distance Weighting regression in a
functional approaching.

For a object-oriented approach, see `globopt.core.regression`.
"""


from typing import Any, Callable, Union

import numpy as np
import numpy.typing as npt
from numba import njit
from typing_extensions import TypeAlias

from globopt.core.regression import (
    RBF_FUNCS,
    IdwRegression,
    RbfRegression,
    _blockwise_inversion,
    _linsolve_via_svd,
    cdist,
    idw_weighting,
    pdist,
    squareform,
)

IdwFitResult: TypeAlias = tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
RbfFitResult: TypeAlias = tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]


def get_fitresult(
    mdl: Union[IdwRegression, RbfRegression]
) -> Union[IdwFitResult, RbfFitResult]:
    """Extracts the fit result from a regression model.

    Parameters
    ----------
    mdl : IdwRegression or RbfRegression]
        The (partially) fitted model to get the result from.

    Returns
    -------
    IdwFitResult or RbfFitResult
        The result of the IDW or RBF regression.
    """
    return (
        (mdl.X_, mdl.y_)
        if isinstance(mdl, IdwRegression)
        else (mdl.X_, mdl.y_, mdl.coef_, mdl.Minv_)
    )


def idw_fit(X: npt.NDArray[np.floating], y: npt.NDArray[np.floating]) -> IdwFitResult:
    """Fits an IDW model to the data.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        The input data to be fitted.
    y : array of shape (n_samples,)
        The target values to be fitted.

    Returns
    -------
    IdwFitResult
        The result of the fit operation for IDW regression.

    Note
    ----
    This is the functional equivalent of
    ```python
    IdwRegressor().fit(X, y)
    ```
    but no input validation is performed here.
    """
    return X, y


@njit
def idw_partial_fit(
    fitresult: IdwFitResult, X: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
) -> IdwFitResult:
    """Fits an already partially fitted IDW model to the additional data.

    Parameters
    ----------
    fitresult : IdwFitResult
        Result of a previous fit or partial fit operation.
    X : array of shape (n_samples, n_features)
        The input data to be fitted.
    y : array of shape (n_samples,)
        The target values to be fitted.

    Returns
    -------
    IdwFitResult
        The newly result of the partial fit operation for IDW regression.

    Note
    ----
    This is the functional equivalent of
    ```python
    IdwRegressor().fit(X_old, y_old).partial_fit(X, y)
    ```
    but no input validation is performed here.
    """
    X_, y_ = fitresult
    return (np.concatenate((X_, X), axis=0), np.concatenate((y_, y), axis=0))


# cannot jit due to idw_weighting's cdist
def idw_predict(
    fitresult: IdwFitResult, X: npt.NDArray[np.floating], exp_weighting: bool = False
) -> npt.NDArray[np.floating]:
    """Predicts target values according to the IDW model.

    Parameters
    ----------
    fitresult : IdwFitResult
        Result of a previous fit or partial fit operation.
    X : array of shape (n_samples, n_features)
        The input data for which `y` has to be predicted.
    exp_weighting : bool, optional
        Whether the weighting function should decay exponentially, by default `False`.

    Returns
    -------
    y: array of floats
        Prediction of `y`.

    Note
    ----
    This is the functional equivalent of
    ```python
    IdwRegressor(exp_weighting).fit(X_old, y_old).predict(X)
    ```
    but no input validation is performed here.
    """
    X_, y_ = fitresult
    W = idw_weighting(X, X_, exp_weighting)
    v = W / W.sum(axis=0)
    return v.T @ y_


# cannot jit due to pdist
def rbf_fit(
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    kernel: str = "inversequadratic",
    eps: float = 1.0775,
) -> RbfFitResult:
    """Fits an RBF model to the data.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        The input data to be fitted.
    y : array of shape (n_samples,)
        The target values to be fitted.
    kernel : {'inversequadratic', 'multiquadric', 'linear', 'gaussian',
        'thinplatespline', 'inversemultiquadric' }
        The type of RBF kernel to use.
    eps : float, optional
        Distance-scaling parameter for the RBF kernel, by default `1.0775`.

    Returns
    -------
    RbfFitResult
        The result of the fit operation for RBF regression.

    Note
    ----
    This is the functional equivalent of
    ```python
    RbfRegressor(kernel, eps).fit(X, y)
    ```
    but no input validation is performed here.
    """
    fun = RBF_FUNCS[kernel]
    d2 = pdist(X, "sqeuclidean")
    M = squareform(fun(d2, eps))
    M[np.diag_indices_from(M)] = fun(0, eps)
    coef_, Minv_ = _linsolve_via_svd(M, y)
    return (X, y, coef_, Minv_)


# cannot jit due to cdist and pdist
def rbf_partial_fit(
    fitresult: RbfFitResult,
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    kernel: str = "inversequadratic",
    eps: float = 1.0775,
) -> RbfFitResult:
    """Fits an already partially fitted RBF model to the additional data.

    Parameters
    ----------
    fitresult : RbfFitResult
        Result of a previous fit or partial fit operation.
    X : array of shape (n_samples, n_features)
        The input data to be fitted.
    y : array of shape (n_samples,)
        The target values to be fitted.
    kernel : {'inversequadratic', 'multiquadric', 'linear', 'gaussian',
        'thinplatespline', 'inversemultiquadric' }
        The type of RBF kernel to use.
    eps : float, optional
        Distance-scaling parameter for the RBF kernel, by default `1.0775`.

    Returns
    -------
    RbfFitResult
        The newly result of the partial fit operation for RBF regression.

    Note
    ----
    This is the functional equivalent of
    ```python
    RbfRegressor(kernel, eps).fit(X_old, y_old).partial_fit(X, y)
    ```
    but no input validation is performed here.
    """
    X_, y_, _, Minv_ = fitresult
    fun: Callable[[Any, float], np.ndarray] = RBF_FUNCS[kernel]
    Phi = fun(cdist(X_, X, "sqeuclidean"), eps)
    phi = pdist(X, "sqeuclidean")
    phi = squareform(fun(phi, eps))
    phi[np.diag_indices_from(phi)] = fun(0, eps)
    y_new, coef_new, Minv_new = _blockwise_inversion(y_, y, Minv_, phi, Phi)
    X_new = np.concatenate((X_, X), axis=0)
    return (X_new, y_new, coef_new, Minv_new)


# cannot jit due to cdist
def rbf_predict(
    fitresult: RbfFitResult,
    X: npt.NDArray[np.floating],
    kernel: str = "inversequadratic",
    eps: float = 1.0775,
) -> npt.NDArray[np.floating]:
    """Predicts target values according to the IDW model.

    Parameters
    ----------
    fitresult : RbfFitResult
        Result of a previous fit or partial fit operation.
    X : array of shape (n_samples, n_features)
        The input data for which `y` has to be predicted.
    kernel : {'inversequadratic', 'multiquadric', 'linear', 'gaussian',
        'thinplatespline', 'inversemultiquadric' }
        The type of RBF kernel to use.
    eps : float, optional
        Distance-scaling parameter for the RBF kernel, by default `1.0775`.

    Returns
    -------
    y: array of floats
        Prediction of `y`.

    Note
    ----
    This is the functional equivalent of
    ```python
    RbfRegressor(kernel, eps).fit(X_old, y_old).predict(X)
    ```
    but no input validation is performed here.
    """
    X_, coef_ = fitresult[0], fitresult[2]
    d2 = cdist(X_, X, "sqeuclidean")
    M = RBF_FUNCS[kernel](d2, eps)
    return M.T @ coef_
