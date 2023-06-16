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


from typing import Literal, NamedTuple, Union

import numba as nb
import numpy as np
from typing_extensions import TypeAlias
from vpso.jit import _float, jit
from vpso.math import batch_cdist, batch_cdist_and_pdist, batch_pdist
from vpso.typing import Array2d, Array3d

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
    svd_tol: float = 1e-9
    exp_weighting: bool = False

    Xm_: Array3d = np.empty((0, 0, 0))
    ym_: Array3d = np.empty((0, 0, 0))
    coef_: Array3d = np.empty((0, 0, 0))
    Minv_: Array3d = np.empty((0, 0, 0))

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

    Xm_: Array3d = np.empty((0, 0, 0))
    ym_: Array3d = np.empty((0, 0, 0))

    def __str__(self) -> str:
        return _regressor_to_str(self)

    def __repr__(self) -> str:
        return self.__str__()


RegressorType: TypeAlias = Union[Rbf, Idw]


"""Small value to avoid division by zero."""
DELTA = 1e-12


@jit(
    [
        _float(_float, _float, nb.types.unicode_type),
        _float[:, :](_float[:, :], _float, nb.types.unicode_type),
        _float[:, :, :](_float[:, :, :], _float, nb.types.unicode_type),
    ],
)
def rbf(d2: np.ndarray, eps: float, type: str) -> np.ndarray:
    if type == "inversequadratic":
        return 1 / (1 + eps**2 * d2)
    if type == "multiquadric":
        return np.sqrt(1 + eps**2 * d2)
    if type == "linear":
        return eps * np.sqrt(d2)
    if type == "gaussian":
        return np.exp(-(eps**2) * d2)
    if type == "thinplatespline":
        return eps**2 * d2 * np.log(np.maximum(eps * d2, DELTA))
    if type == "inversemultiquadric":
        return 1 / np.sqrt(1 + eps**2 * d2)
    raise ValueError(f"unknown RBF kernel type: {type}")


@jit(
    nb.types.UniTuple(_float[:, :, :], 2)(
        _float[:, :, :], _float[:, :, :], _float, nb.types.unicode_type, _float
    ),
    parallel=True,
)
def _fit_rbf_via_svd(
    X: Array3d, y: Array3d, eps: float, kernel: str, svd_tol: float
) -> tuple[Array3d, Array3d]:
    """Fits the RBF to the data by solving the linear systems via SVD."""
    M = rbf(batch_pdist(X, "sqeuclidean"), eps, kernel)
    B, n, _ = y.shape
    coef = np.empty((B, n, 1), dtype=np.float64)
    Minv = np.empty((B, n, n), dtype=np.float64)
    for i in nb.prange(B):
        U, S, VT = np.linalg.svd(M[i])
        S[S <= svd_tol] = np.inf
        Minv_ = (VT.T / S) @ U.T
        Minv[i] = Minv_
        coef[i] = Minv_ @ y[i]
    return coef, Minv


def _rbf_fit(mdl: Rbf, X: Array3d, y: Array3d) -> Rbf:
    """Fits an RBF model to the data."""
    # create matrix of kernel evaluations
    kernel, eps, svd_tol, exp_weighting = mdl[:4]
    coef, Minv = _fit_rbf_via_svd(X, y, eps, kernel, svd_tol)
    return Rbf(kernel, eps, svd_tol, exp_weighting, X, y, coef, Minv)


@jit(
    nb.types.UniTuple(_float[:, :, :], 4)(
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float,
        nb.types.unicode_type,
        _float,
    ),
    parallel=True,
)
def _partial_fit_via_blockwise_inversion(
    Xm: Array3d,
    ym: Array3d,
    X: Array3d,
    y: Array3d,
    Minv: Array3d,
    eps: float,
    kernel: str,
    inv_tol: float,
) -> tuple[Array3d, Array3d, Array3d, Array3d]:
    """Performs blockwise inversion updates of the RBF kernel matrices."""
    # create matrix of kernel evals of new elements w.r.t. training data and themselves
    Phi_and_phi = batch_cdist_and_pdist(X, Xm, "sqeuclidean")
    Phi = rbf(Phi_and_phi[0].transpose(0, 2, 1), eps, kernel)
    phi = rbf(Phi_and_phi[1], eps, kernel)

    # update data
    X_new = np.concatenate((Xm, X), 1)
    y_new = np.concatenate((ym, y), 1)

    # update inverse blockwise
    B, n, _ = ym.shape
    n += y.shape[1]
    coef_new = np.empty((B, n, 1), dtype=np.float64)
    Minv_new = np.empty((B, n, n), dtype=np.float64)
    for i in nb.prange(B):
        Minv_ = Minv[i]
        Phi_ = Phi[i]
        L = Minv_ @ Phi_
        c = phi[i] - Phi_.T @ L
        c = np.where(np.abs(c) <= inv_tol, inv_tol, c)
        c_inv = np.linalg.inv(c)
        B = -L @ c_inv
        A = Minv[i] - B @ L.T
        Minv_new_ = np.concatenate(
            (np.concatenate((A, B), 1), np.concatenate((B.T, c_inv), 1)), 0
        )
        Minv_new[i] = Minv_new_
        coef_new[i] = Minv_new_ @ y_new[i]
    return X_new, y_new, coef_new, Minv_new


def _rbf_partial_fit(mdl: Rbf, X: Array3d, y: Array2d) -> Rbf:
    """Fits an already partially fitted RBF model to the additional data."""
    kernel, eps, svd_tol, exp_weighting, Xm, ym, _, Minv = mdl
    X_new, y_new, coef_new, Minv_new = _partial_fit_via_blockwise_inversion(
        Xm, ym, X, y, Minv, eps, kernel, svd_tol
    )
    return Rbf(kernel, eps, svd_tol, exp_weighting, X_new, y_new, coef_new, Minv_new)


@jit(_float[:, :, :](_float[:, :, :], _float[:, :, :], _float, nb.types.unicode_type))
def _get_rbf_matrix(X: Array3d, Xm: Array3d, eps: float, kernel: str) -> Array3d:
    d2 = batch_cdist(X, Xm, "sqeuclidean")
    return rbf(d2, eps, kernel)


def _rbf_predict(mdl: Rbf, X: Array3d) -> Array2d:
    kernel, eps = mdl[:2]
    """Predicts target values according to the IDW model."""
    M = _get_rbf_matrix(X, mdl.Xm_, eps, kernel)
    return M @ mdl.coef_


@jit(_float[:, :, :](_float[:, :, :], _float[:, :, :], nb.boolean))
def _idw_weighting(X: Array3d, Xm: Array3d, exp_weighting: bool = False) -> Array3d:
    """Computes the IDW weighting function `w`."""
    d2 = batch_cdist(X, Xm, "sqeuclidean")
    W = 1 / np.maximum(d2, DELTA)
    if exp_weighting:
        W *= np.exp(-d2)
    return W


def _idw_fit(mdl: Idw, X: Array3d, y: Array3d) -> Idw:
    """Fits an IDW model to the data."""
    return Idw(mdl.exp_weighting, X, y)


def _idw_partial_fit(mdl: Idw, X: Array3d, y: Array3d) -> Idw:
    """Fits an already partially fitted IDW model to the additional data."""
    exp_weighting, X_, y_ = mdl
    return Idw(exp_weighting, np.concatenate((X_, X), 1), np.concatenate((y_, y), 1))


@jit(_float[:, :, :](_float[:, :, :], _float[:, :, :], nb.boolean))
def _idw_contributions(X: Array3d, Xm: Array3d, exp_weighting: bool) -> Array3d:
    """Computes the IDW contributions `v`."""
    W = _idw_weighting(X, Xm, exp_weighting)
    return W / W.sum(2)[:, :, np.newaxis]


def _idw_predict(mdl: Idw, X: Array3d) -> Array3d:
    """Predicts target values according to the IDW model."""
    exp_weighting, X_, y_ = mdl
    v = _idw_contributions(X, X_, exp_weighting)
    return v @ y_  # cannot be jitted due to 3D tensor multiplication


def fit(mdl: RegressorType, X: Array3d, y: Array3d) -> RegressorType:
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


def partial_fit(mdl: RegressorType, X: Array3d, y: Array3d) -> RegressorType:
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


def predict(mdl: RegressorType, X: Array3d) -> Array3d:
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
    batch.

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
