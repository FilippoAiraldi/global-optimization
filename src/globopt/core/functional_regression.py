"""
Implementation of Radial Basis Function and Inverse Distance Weighting regression in a
functional approaching.

For a object-oriented approach, see `globopt.core.regression`.
"""


from typing import Callable, Literal, Union

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.spatial.distance import cdist, pdist, squareform
from typing_extensions import TypeAlias

IdwFitResult: TypeAlias = tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
RbfFitResult: TypeAlias = tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]
RbfKernel: TypeAlias = Literal[
    "inversequadratic",
    "multiquadric",
    "linear",
    "gaussian",
    "thinplatespline",
    "inversemultiquadric",
]


"""Small value to avoid division by zero."""
DELTA = 1e-9


# no jit here, it's already fast enough
def idw_weighting(
    X: npt.ArrayLike, Xm: npt.ArrayLike, exp_weighting: bool = False
) -> npt.NDArray[np.floating]:
    """Computes the IDW weighting function.

    Parameters
    ----------
    X : array_like
        Array of `x` for which to compute the weighting function.
    Xm : array_like
        Array of observed query points.
    exp_weighting : bool, optional
        Whether the weighting function should decay exponentially, by default `False`.

    Returns
    -------
    array
        The weighiing function computed at `X` against dataset `Xm`.
    """
    d2 = cdist(Xm, X, "sqeuclidean")
    W = 1 / (d2 + DELTA)
    if exp_weighting:
        W *= np.exp(-d2)
    return W


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
    W = idw_weighting(X, X_, exp_weighting)  # create matrix of inverse-distance weights
    v = W / W.sum(axis=0)
    return v.T @ y_  # predict as weighted average based on normalized distance


"""RBF kernels."""
RBF_FUNCS: dict[
    RbfKernel, Callable[[Union[float, np.ndarray], float], Union[float, np.ndarray]]
] = {
    "inversequadratic": lambda d2, eps: 1 / (1 + eps**2 * d2),
    "multiquadric": lambda d2, eps: np.sqrt(1 + eps**2 * d2),
    "linear": lambda d2, eps: eps * np.sqrt(d2),
    "gaussian": lambda d2, eps: np.exp(-(eps**2) * d2),
    "thinplatespline": lambda d2, eps: eps**2 * d2 * np.log(eps * d2 + DELTA),
    "inversemultiquadric": lambda d2, eps: 1 / np.sqrt(1 + eps**2 * d2),
}


@njit
def _linsolve_via_svd(M, y):
    """Internal jit function to solve linear system via SVD."""
    U, S, VT = np.linalg.svd(M)
    Sinv = np.diag(1 / S)
    Minv = VT.T @ Sinv @ U.T
    coef = Minv @ y
    return coef, Minv


@njit
def _blockwise_inversion(ym, y, Minv, phi, Phi):
    """Internal jit function to perform blockwise inversion."""
    L = Minv @ Phi
    c = np.linalg.inv(phi - Phi.T @ L)
    B = -L @ c
    A = Minv - B @ L.T
    Minv_new = np.vstack((np.hstack((A, B)), np.hstack((B.T, c))))

    # update coefficients
    y_new = np.concatenate((ym, y), axis=0)
    coef_new = Minv_new @ y_new
    return y_new, coef_new, Minv_new


# cannot jit due to pdist
def rbf_fit(
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    kernel: RbfKernel = "inversequadratic",
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
    # create matrix of kernel evaluations
    fun = RBF_FUNCS[kernel]
    d2 = pdist(X, "sqeuclidean")  # returns all single distances, not a matrix
    M = squareform(fun(d2, eps))
    M[np.diag_indices_from(M)] = fun(0, eps)

    # compute coefficients via SVD and inverse of M (useful for partial_fit)
    coef_, Minv_ = _linsolve_via_svd(M, y)
    return (X, y, coef_, Minv_)


# cannot jit due to cdist and pdist
def rbf_partial_fit(
    fitresult: RbfFitResult,
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    kernel: RbfKernel = "inversequadratic",
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

    # create matrix of kernel evals of new elements w.r.t. training data and themselves
    fun = RBF_FUNCS[kernel]
    Phi = fun(cdist(X_, X, "sqeuclidean"), eps)
    phi = pdist(X, "sqeuclidean")
    phi = squareform(fun(phi, eps))
    phi[np.diag_indices_from(phi)] = fun(0, eps)

    # update inverse of M via blockwise inversion and coefficients
    y_new, coef_new, Minv_new = _blockwise_inversion(y_, y, Minv_, phi, Phi)
    X_new = np.concatenate((X_, X), axis=0)
    return (X_new, y_new, coef_new, Minv_new)


# cannot jit due to cdist
def rbf_predict(
    fitresult: RbfFitResult,
    X: npt.NDArray[np.floating],
    kernel: RbfKernel = "inversequadratic",
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

    # create matrix of kernel evaluations
    d2 = cdist(X_, X, "sqeuclidean")
    M = RBF_FUNCS[kernel](d2, eps)

    # predict as linear combination
    return M.T @ coef_  # type: ignore[union-attr]
