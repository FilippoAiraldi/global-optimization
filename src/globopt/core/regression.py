"""
Implementation of Radial Basis Function and Inverse Distance Weighting regression
according to [1]. These regression models are coded according to a functional approach
rather than object-oriented. Still, the common interface with `fit`, `partial_fit` and
`predict` is offered.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Callable, Literal, NamedTuple, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist, pdist, squareform
from typing_extensions import TypeAlias

Array: TypeAlias = npt.NDArray[np.floating]


"""Choices of available RBF kernels."""
RbfKernel: TypeAlias = Literal[
    "inversequadratic",
    "multiquadric",
    "linear",
    "gaussian",
    "thinplatespline",
    "inversemultiquadric",
]


def _regressor_to_str(regressor: NamedTuple) -> str:
    """Return a string representation of a regressor."""

    def _params_to_str():
        for param in regressor._fields:
            if not param.endswith("_"):
                val = getattr(regressor, param)
                if val != regressor._field_defaults[param]:
                    yield f"{param}={val}"

    return f"{regressor.__class__.__name__}(" + ", ".join(_params_to_str()) + ")"


class Rbf(NamedTuple):
    """Options for RBF regression in Global Optimization.

    Parameters
    ----------
    kernel : {'inversequadratic', 'multiquadric', 'linear', 'gaussian',
        'thinplatespline', 'inversemultiquadric' }
        The type of RBF kernel to use.
    eps : float, optional
        Distance-scaling parameter for the RBF kernel, by default `1.0775`.
    svd_tol : float, optional
        Tolerance for the singular value decomposition for inversion, by default `1e-6`.
    exp_weighting : bool, optional
        Whether the weighting function should decay exponentially, by default `False`.
        This option is only used during the computation of the acquisition function but
        not during regression.
    """

    kernel: RbfKernel = "inversequadratic"
    eps: float = 1.0775
    svd_tol: float = 1e-6
    exp_weighting: bool = False

    Xm_: Array = np.empty((0, 0))
    ym_: Array = np.empty((0,))
    coef_: Array = np.empty((0,))
    Minv_: Array = np.empty((0, 0))

    def __str__(self) -> str:
        return _regressor_to_str(self)

    def __repr__(self) -> str:
        return self.__str__()


class Idw(NamedTuple):
    """Options for IDW regression in Global Optimization.

    Parameters
    ----------
    exp_weighting : bool, optional
        Whether the weighting function should decay exponentially, by default `False`.
    """

    exp_weighting: bool = False

    Xm_: Array = np.empty((0, 0))
    ym_: Array = np.empty((0,))

    def __str__(self) -> str:
        return _regressor_to_str(self)

    def __repr__(self) -> str:
        return self.__str__()


RegressorType: TypeAlias = Union[Rbf, Idw]


"""Small value to avoid division by zero."""
DELTA = 1e-9


"""RBF kernels."""
RBF_FUNCS: dict[RbfKernel, Callable[[Array, float], Array]] = {
    "inversequadratic": lambda d2, eps: 1 / (1 + eps**2 * d2),
    "multiquadric": lambda d2, eps: np.sqrt(1 + eps**2 * d2),
    "linear": lambda d2, eps: eps * np.sqrt(d2),
    "gaussian": lambda d2, eps: np.exp(-(eps**2) * d2),
    "thinplatespline": lambda d2, eps: eps**2
    * d2
    * np.log(np.maximum(eps * d2, DELTA)),
    "inversemultiquadric": lambda d2, eps: 1 / np.sqrt(1 + eps**2 * d2),
}


def _linsolve_via_svd(M: Array, y: Array, tol: float = 1e-6) -> tuple[Array, Array]:
    """Solves linear systems via SVD."""
    U, S, VT = np.linalg.svd(M)
    S[S <= tol] = np.inf
    Minv = (VT.T / S.reshape(1, -1)) @ U.T
    coef = Minv @ y
    return coef, Minv


def _blockwise_inversion(
    ym: Array, y: Array, Minv: Array, phi: Array, Phi: Array, tol: float = 1e-6
) -> tuple[Array, Array, Array]:
    """Performs blockwise inversion updates of RBF kernel matrices."""
    L = Minv @ Phi
    #
    c = phi - Phi.T @ L
    c[np.abs(c) <= tol] = tol
    c_inv = np.linalg.inv(c)
    #
    B = -L @ c_inv
    A = Minv - B @ L.T
    Minv_new = np.vstack((np.hstack((A, B)), np.hstack((B.T, c_inv))))

    # update coefficients
    y_new = np.concatenate((ym, y), 0)
    coef_new = Minv_new @ y_new
    return y_new, coef_new, Minv_new


def idw_weighting(X: Array, Xm: Array, exp_weighting: bool = False) -> Array:
    """Computes the IDW weighting function.

    Parameters
    ----------
    X : array of shape (n_target, n_features)
        Array of `x` for which to compute the weighting function.
    Xm : array of shape (n_samples, n_features)
        Array of observed query points.
    exp_weighting : bool, optional
        Whether the weighting function should decay exponentially, by default `False`.

    Returns
    -------
    array
        The weighiing function computed at `X` against dataset `Xm`.
    """
    d2 = cdist(X, Xm, "sqeuclidean")
    W = 1 / np.maximum(d2, DELTA)
    if exp_weighting:
        W *= np.exp(-d2)
    return W


def _idw_fit(mdl: Idw, X: Array, y: Array) -> Idw:
    """Fits an IDW model to the data."""
    return Idw(mdl.exp_weighting, X, y)


def _idw_partial_fit(mdl: Idw, X: Array, y: Array) -> Idw:
    """Fits an already partially fitted IDW model to the additional data."""
    exp_weighting, X_, y_ = mdl
    return Idw(exp_weighting, np.concatenate((X_, X), 0), np.concatenate((y_, y), 0))


def _idw_predict(mdl: Idw, X: Array) -> Array:
    """Predicts target values according to the IDW model."""
    exp_weighting, X_, y_ = mdl
    W = idw_weighting(X, X_, exp_weighting)
    v = W / W.sum(1, keepdims=True)
    return v @ y_


def _rbf_fit(mdl: Rbf, X: Array, y: Array) -> Rbf:
    """Fits an RBF model to the data."""
    # create matrix of kernel evaluations
    kernel, eps, svd_tol, exp_weighting = mdl[:4]
    fun = RBF_FUNCS[kernel]
    d2 = pdist(X, "sqeuclidean")  # returns all single distances, not a matrix
    M = squareform(fun(d2, eps))
    M[np.diag_indices_from(M)] = fun(0, eps)  # type: ignore[arg-type]

    # compute coefficients via SVD and inverse of M (useful for partial_fit)
    coef, Minv = _linsolve_via_svd(M, y, svd_tol)
    return Rbf(kernel, eps, svd_tol, exp_weighting, X, y, coef, Minv)


def _rbf_partial_fit(mdl: Rbf, X: Array, y: Array) -> Rbf:
    """Fits an already partially fitted RBF model to the additional data."""
    kernel, eps, svd_tol, exp_weighting, Xm, ym, _, Minv = mdl

    # create matrix of kernel evals of new elements w.r.t. training data and themselves
    fun = RBF_FUNCS[kernel]
    Phi = fun(cdist(Xm, X, "sqeuclidean"), eps)
    phi = pdist(X, "sqeuclidean")
    phi = squareform(fun(phi, eps))
    phi[np.diag_indices_from(phi)] = fun(0, eps)  # type: ignore[arg-type]

    # update inverse of M via blockwise inversion and coefficients
    y_new, coef_new, Minv_new = _blockwise_inversion(ym, y, Minv, phi, Phi, svd_tol)
    X_new = np.concatenate((Xm, X), 0)
    return Rbf(kernel, eps, svd_tol, exp_weighting, X_new, y_new, coef_new, Minv_new)


def _rbf_predict(mdl: Rbf, X: Array) -> Array:
    """Predicts target values according to the IDW model."""
    d2 = cdist(mdl.Xm_, X, "sqeuclidean")
    M = RBF_FUNCS[mdl.kernel](d2, mdl.eps)
    return M.T @ mdl.coef_


def fit(mdl: RegressorType, X: Array, y: Array) -> RegressorType:
    """Fits an IDW or RBF model to the data.

    Parameters
    ----------
    mdl : Idw or Rbf
        The options for the RBF model.
    X : array of shape (n_samples, n_features)
        The input data to be fitted.
    y : array of shape (n_samples,)
        The target values to be fitted.

    Returns
    -------
    Idw or Rbf
        The result of the fit operation for RBF regression.
    """
    return _rbf_fit(mdl, X, y) if isinstance(mdl, Rbf) else _idw_fit(mdl, X, y)


def partial_fit(mdl: RegressorType, X: Array, y: Array) -> RegressorType:
    """Adds additional data to an already fitted IDW or RBF model.

    Parameters
    ----------
    fitresult : Idw or Rbf
        Model resulting from a previous fit or partial fit operation.
    X : array of shape (n_samples, n_features)
        The input data to be fitted.
    y : array of shape (n_samples,)
        The target values to be fitted.

    Returns
    -------
    Idw or Rbf
        The newly result of the partial fit operation for the regression.
    """
    return (
        _rbf_partial_fit(mdl, X, y)
        if isinstance(mdl, Rbf)
        else _idw_partial_fit(mdl, X, y)
    )


def predict(mdl: RegressorType, X: Array) -> Array:
    """Predicts target values according to the IDW or RBF model.

    Parameters
    ----------
    mdl : Idw or Rbf
        Model resulting from a previous fit or partial fit operation.
    X : array of shape (n_samples, n_features)
        The input data for which `y` has to be predicted.

    Returns
    -------
    y: array of floats
        Prediction of `y`.
    """
    return _rbf_predict(mdl, X) if isinstance(mdl, Rbf) else _idw_predict(mdl, X)
