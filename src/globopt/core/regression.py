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
from scipy.spatial.distance import (
    _copy_array_if_base_present,
    _distance_pybind,
    _distance_wrap,
)
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

    Xm_: Array = np.empty((0, 0, 0))
    ym_: Array = np.empty((0, 0, 0))
    coef_: Array = np.empty((0, 0, 0))
    Minv_: Array = np.empty((0, 0, 0))

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

    Xm_: Array = np.empty((0, 0, 0))
    ym_: Array = np.empty((0, 0, 0))

    def __str__(self) -> str:
        return _regressor_to_str(self)

    def __repr__(self) -> str:
        return self.__str__()


RegressorType: TypeAlias = Union[Rbf, Idw]


"""Small value to avoid division by zero."""
DELTA = 1e-9


# @njit
# def _batch_sqeuclidean_cdist(X: Array, Y: Array) -> Array:
#     B, Nx, n = X.shape
#     Ny = Y.shape[1]
#     X = X.reshape(B, Nx, 1, n)
#     Y = Y.reshape(B, 1, Ny, n)
#     return np.square(X - Y).sum(-1)


def _batch_sqeuclidean_cdist(X: Array, Y: Array) -> Array:
    """Computes the squared ecludian distance matrices for 3D tensors."""
    # return np.asarray([cdist(X[i], Y[i], "sqeuclidean") for i in range(X.shape[0])])
    B = X.shape[0]
    out = np.empty((B, X.shape[1], Y.shape[1]), dtype=X.dtype)
    for i in range(B):
        _distance_pybind.cdist_sqeuclidean(X[i], Y[i], out=out[i])
    return out


def _batch_sqeuclidean_pdist(X: Array) -> Array:
    """Computes the pairwise squared ecludian distance matrices for 3D tensors."""
    # return np.asarray([pdist(X[i], "sqeuclidean") for i in range(X.shape[0])])
    B, nx = X.shape[:2]
    out = np.empty((B, (nx - 1) * nx // 2), dtype=X.dtype)
    for i in range(B):
        _distance_pybind.pdist_sqeuclidean(X[i], out=out[i])
    return out


def _batch_squareform(D: Array) -> Array:
    """Converts a batch of pairwise distance matrices to distance matrices."""
    # return np.asarray([squareform(D[i]) for i in range(D.shape[0])])
    B, n = D.shape
    d = int(0.5 * (np.sqrt(8 * n + 1) + 1))
    M = np.zeros((B, d, d), dtype=D.dtype)
    D = _copy_array_if_base_present(D)
    for i in range(B):
        _distance_wrap.to_squareform_from_vector_wrap(M[i], D[i])
    return M


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
    B, n, _ = y.shape
    U, S, VT = np.linalg.svd(M)
    #
    S[S <= tol] = np.inf
    Sinv = 1 / S.reshape(B, 1, n)
    #
    Minv = (VT.transpose((0, 2, 1)) * Sinv) @ U.transpose(0, 2, 1)
    coef = Minv @ y
    return coef, Minv


def _blockwise_inversion(
    ym: Array, y: Array, Minv: Array, phi: Array, Phi: Array, tol: float = 1e-6
) -> tuple[Array, Array, Array]:
    """Performs blockwise inversion updates of RBF kernel matrices."""
    L = Minv @ Phi
    #
    c = phi - Phi.transpose(0, 2, 1) @ L
    c[np.abs(c) <= tol] = tol
    c_inv = np.linalg.inv(c)
    #
    B = -L @ c_inv
    A = Minv - B @ L.transpose(0, 2, 1)
    Minv_new = np.concatenate(
        (np.concatenate((A, B), 2), np.concatenate((B.transpose(0, 2, 1), c_inv), 2)), 1
    )

    # update coefficients
    y_new = np.concatenate((ym, y), 1)
    coef_new = Minv_new @ y_new
    return y_new, coef_new, Minv_new


def idw_weighting(X: Array, Xm: Array, exp_weighting: bool = False) -> Array:
    """Computes the IDW weighting function.

    Parameters
    ----------
    X : array of shape (batch, n_target, n_features)
        Array of `x` for which to compute the weighting function.
    Xm : array of shape (batch, n_samples, n_features)
        Array of observed query points.
    exp_weighting : bool, optional
        Whether the weighting function should decay exponentially, by default `False`.

    Returns
    -------
    array
        The weighiing function computed at `X` against dataset `Xm`.
    """
    d2 = _batch_sqeuclidean_cdist(X, Xm)
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
    return Idw(exp_weighting, np.concatenate((X_, X), 1), np.concatenate((y_, y), 1))


def _idw_predict(mdl: Idw, X: Array) -> Array:
    """Predicts target values according to the IDW model."""
    exp_weighting, X_, y_ = mdl
    W = idw_weighting(X, X_, exp_weighting)
    v = W / W.sum(2, keepdims=True)
    return v @ y_


def _rbf_fit(mdl: Rbf, X: Array, y: Array) -> Rbf:
    """Fits an RBF model to the data."""
    # create matrix of kernel evaluations
    kernel, eps, svd_tol, exp_weighting = mdl[:4]
    fun = RBF_FUNCS[kernel]
    d2 = _batch_sqeuclidean_pdist(X)  # returns all single distances, not a matrix
    M = _batch_squareform(fun(d2, eps))
    M[:, np.eye(M.shape[1], dtype=bool)] = fun(0, eps)  # type: ignore[arg-type]

    # compute coefficients via SVD and inverse of M (useful for partial_fit)
    coef, Minv = _linsolve_via_svd(M, y, svd_tol)
    return Rbf(kernel, eps, svd_tol, exp_weighting, X, y, coef, Minv)


def _rbf_partial_fit(mdl: Rbf, X: Array, y: Array) -> Rbf:
    """Fits an already partially fitted RBF model to the additional data."""
    kernel, eps, svd_tol, exp_weighting, Xm, ym, _, Minv = mdl

    # create matrix of kernel evals of new elements w.r.t. training data and themselves
    fun = RBF_FUNCS[kernel]
    Phi = fun(_batch_sqeuclidean_cdist(Xm, X), eps)
    phi = _batch_sqeuclidean_pdist(X)
    phi = _batch_squareform(fun(phi, eps))
    phi[:, np.eye(phi.shape[1], dtype=bool)] = fun(0, eps)  # type: ignore[arg-type]

    # update inverse of M via blockwise inversion and coefficients
    y_new, coef_new, Minv_new = _blockwise_inversion(ym, y, Minv, phi, Phi, svd_tol)
    X_new = np.concatenate((Xm, X), 1)
    return Rbf(kernel, eps, svd_tol, exp_weighting, X_new, y_new, coef_new, Minv_new)


def _rbf_predict(mdl: Rbf, X: Array) -> Array:
    """Predicts target values according to the IDW model."""
    d2 = _batch_sqeuclidean_cdist(mdl.Xm_, X)
    M = RBF_FUNCS[mdl.kernel](d2, mdl.eps)
    return M.transpose(0, 2, 1) @ mdl.coef_


def fit(mdl: RegressorType, X: Array, y: Array) -> RegressorType:
    """Fits an IDW or RBF model to the data.

    Parameters
    ----------
    mdl : Idw or Rbf
        The options for the RBF model.
    X : array of shape (batch, n_samples, n_features)
        The input data to be fitted.
    y : array of shape (batch, n_samples, 1)
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
    X : array of shape (batch, n_samples, n_features)
        The input data to be fitted.
    y : array of shape (batch, n_samples, 1)
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
    X : array of shape (batch, n_samples, n_features)
        The input data for which `y` has to be predicted.

    Returns
    -------
    y: array of floats
        Prediction of `y`.
    """
    return _rbf_predict(mdl, X) if isinstance(mdl, Rbf) else _idw_predict(mdl, X)


def repeat(mdl: RegressorType, n: int) -> RegressorType:
    """Repeats a regressor model `n` times, so that `n` regressions can be computed in
    parallel per batch.

    Parameters
    ----------
    mdl : RegressorType
        The regressor model to be repeated.
    n : int
        The number of times the regressor model should be repeated.

    Returns
    -------
    RegressorType
        The repeated regressor model. The repetitions are all the same model.
    """
    if isinstance(mdl, Idw):
        return Idw(mdl.exp_weighting, mdl.Xm_.repeat(n, 0), mdl.ym_.repeat(n, 0))
    else:
        return Rbf(
            mdl.kernel,
            mdl.eps,
            mdl.svd_tol,
            mdl.exp_weighting,
            mdl.Xm_.repeat(n, 0),
            mdl.ym_.repeat(n, 0),
            mdl.coef_.repeat(n, 0),
            mdl.Minv_.repeat(n, 0),
        )
