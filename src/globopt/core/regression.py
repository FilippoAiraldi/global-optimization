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


from enum import IntEnum
from typing import NamedTuple, Union

import numba as nb
import numpy as np
from typing_extensions import TypeAlias
from vpso.math import batch_cdist, batch_cdist_and_pdist, batch_pdist
from vpso.typing import Array3d

"""Small value to avoid division by zero."""
DELTA = 1e-12


class Kernel(IntEnum):
    """Kernels for RBF regression."""

    InverseQuadratic = 0
    Multiquadric = 1
    Linear = 2
    Gaussian = 3
    ThinPlateSpline = 4
    InverseMultiquadric = 5


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

    kernel: Kernel = Kernel.InverseQuadratic
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
nb_Kernel = nb.types.IntEnumMember(Kernel, nb.int64)
nb_Rbf = nb.types.NamedTuple(
    (
        nb_Kernel,
        nb.float64,
        nb.float64,
        nb.bool_,
        nb.float64[:, :, ::1],
        nb.float64[:, :, ::1],
        nb.float64[:, :, ::1],
        nb.float64[:, :, ::1],
    ),
    Rbf,
)
nb_Idw = nb.types.NamedTuple(
    (nb.bool_, nb.float64[:, :, ::1], nb.float64[:, :, ::1]), Idw
)


@nb.njit(
    nb.float64[:, :, :](nb.float64[:, :, :], nb.float64[:, :, :]),
    cache=True,
    nogil=True,
    parallel=True,
)
def matmul3d(X: Array3d, Y: Array3d) -> Array3d:
    """Performs matrix multiplication between two 3d arrays."""
    B, M, _ = X.shape
    N = Y.shape[2]
    out = np.empty((B, M, N), dtype=X.dtype)
    for i in nb.prange(B):
        np.dot(X[i], Y[i], out=out[i])
    return out


@nb.njit(
    [
        nb.float64(nb.float64, nb.float64, nb_Kernel),
        nb.float64[:](nb.float64[:], nb.float64, nb_Kernel),
        nb.float64[:, :](nb.float64[:, :], nb.float64, nb_Kernel),
        nb.float64[:, :, :](nb.float64[:, :, :], nb.float64, nb_Kernel),
    ],
    cache=True,
    nogil=True,
)
def rbf(d2: np.ndarray, eps: float, kernel: Kernel) -> np.ndarray:
    if kernel == Kernel.InverseQuadratic.value:
        return 1 / (1 + eps**2 * d2)
    if kernel == Kernel.Multiquadric.value:
        return np.sqrt(1 + eps**2 * d2)
    if kernel == Kernel.Linear.value:
        return eps * np.sqrt(d2)
    if kernel == Kernel.Gaussian.value:
        return np.exp(-(eps**2) * d2)
    if kernel == Kernel.ThinPlateSpline.value:
        return eps**2 * d2 * np.log(np.maximum(eps * d2, DELTA))
    if kernel == Kernel.InverseMultiquadric.value:
        return 1 / np.sqrt(1 + eps**2 * d2)
    raise ValueError(f"unknown RBF kernel: {kernel}")


@nb.njit(
    nb_Rbf(nb_Rbf, nb.float64[:, :, :], nb.float64[:, :, :]),
    cache=True,
    nogil=True,
    parallel=True,
)
def _rbf_fit(mdl: Rbf, X: Array3d, y: Array3d) -> Rbf:
    """Fits an RBF model to the data (via SVD)."""
    kernel, eps, svd_tol, exp_weighting = mdl[:4]
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)

    B, n, _ = y.shape
    M = rbf(batch_pdist(X, "sqeuclidean"), eps, kernel)
    coef = np.empty((B, n, 1), dtype=np.float64)
    Minv = np.empty((B, n, n), dtype=np.float64)
    for i in nb.prange(B):
        U, S, VT = np.linalg.svd(M[i])
        S[S <= svd_tol] = np.inf
        Minv_ = (VT.T / S) @ U.T
        Minv[i] = Minv_
        coef[i] = Minv_ @ y[i]

    return Rbf(kernel, eps, svd_tol, exp_weighting, X, y, coef, Minv)


@nb.njit(
    nb_Rbf(nb_Rbf, nb.float64[:, :, :], nb.float64[:, :, :]),
    cache=True,
    nogil=True,
    parallel=True,
)
def _rbf_partial_fit(mdl: Rbf, X: Array3d, y: Array3d) -> Rbf:
    """Fits an already partially fitted RBF model to the additional data (via blockwise
    inversion)."""
    kernel, eps, svd_tol, exp_weighting, Xm, ym, _, Minv = mdl

    X_new = np.ascontiguousarray(np.concatenate((Xm, X), 1))
    y_new = np.ascontiguousarray(np.concatenate((ym, y), 1))
    N, n, _ = y_new.shape
    coef_new = np.empty((N, n, 1), dtype=np.float64)
    Minv_new = np.empty((N, n, n), dtype=np.float64)
    Phi_and_phi = batch_cdist_and_pdist(X, Xm, "sqeuclidean")
    Phi = rbf(Phi_and_phi[0].transpose(0, 2, 1), eps, kernel)
    phi = rbf(Phi_and_phi[1], eps, kernel)
    for i in nb.prange(N):
        Minv_ = Minv[i]
        Phi_ = Phi[i]
        L = Minv_ @ Phi_
        c = phi[i] - Phi_.T @ L
        c = np.where(np.abs(c) <= svd_tol, svd_tol, c)
        c_inv = np.linalg.inv(c)
        B = -L @ c_inv
        A = Minv[i] - B @ L.T
        Minv_new_ = np.concatenate(
            (np.concatenate((A, B), 1), np.concatenate((B.T, c_inv), 1)), 0
        )
        Minv_new[i] = Minv_new_
        coef_new[i] = Minv_new_ @ y_new[i]

    return Rbf(kernel, eps, svd_tol, exp_weighting, X_new, y_new, coef_new, Minv_new)


@nb.njit(nb.float64[:, :, :](nb.float64[:, :, :], nb.float64[:, :, :], nb.bool_))
def _idw_weighting(X: Array3d, Xm: Array3d, exp_weighting: bool = False) -> Array3d:
    """Computes the IDW weighting function `w`."""
    d2 = batch_cdist(X, Xm, "sqeuclidean")
    W = 1 / np.maximum(d2, DELTA)
    if exp_weighting:
        W *= np.exp(-d2)
    return W


@nb.njit(
    [
        nb_Idw(nb_Idw, nb.float64[:, :, :], nb.float64[:, :, :]),
        nb_Rbf(nb_Rbf, nb.float64[:, :, :], nb.float64[:, :, :]),
    ],
    cache=True,
    nogil=True,
)
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
    return (
        Idw(mdl.exp_weighting, np.ascontiguousarray(X), np.ascontiguousarray(y))
        if len(mdl) == 3
        else _rbf_fit(mdl, X, y)
    )


@nb.njit(
    [
        nb_Idw(nb_Idw, nb.float64[:, :, :], nb.float64[:, :, :]),
        nb_Rbf(nb_Rbf, nb.float64[:, :, :], nb.float64[:, :, :]),
    ],
    cache=True,
    nogil=True,
)
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
        Idw(
            mdl.exp_weighting,
            np.ascontiguousarray(np.concatenate((mdl.Xm_, X), 1)),
            np.ascontiguousarray(np.concatenate((mdl.ym_, y), 1)),
        )
        if len(mdl) == 3
        else _rbf_partial_fit(mdl, X, y)
    )


@nb.njit(
    [
        nb.float64[:, :, :](nb_Idw, nb.float64[:, :, :]),
        nb.float64[:, :, :](nb_Rbf, nb.float64[:, :, :]),
    ],
    cache=True,
    nogil=True,
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
    if len(mdl) == 3:  # IDW
        exp_weighting, X_, y_ = mdl  # type: ignore[misc]
        W = _idw_weighting(X, X_, exp_weighting)
        v = W / W.sum(2)[:, :, np.newaxis]
        return matmul3d(v, y_)

    M = rbf(batch_cdist(X, mdl.Xm_, "sqeuclidean"), mdl.eps, mdl.kernel)  # type: ignore
    return matmul3d(M, mdl.coef_)  # type: ignore[union-attr]


@nb.njit(
    [
        nb.float64[:, :](nb.float64[:, :], nb.int64),
        nb.float64[:, :, :](nb.float64[:, :, :], nb.int64),
    ],
    cache=True,
    nogil=True,
)
def repeat_along_first_axis(x: Array3d, n: int) -> Array3d:
    """Repeats an array `n` times along axis=0. The array must be either 2D or 3D, and
    the size of the input array on axis=0 must be 1."""
    if x.shape[0] != 1:
        raise ValueError("first dimension of x must be 1.")
    if x.ndim == 2:
        M = x.shape[1]
        return x.repeat(n).reshape(M, n).T
    if x.ndim == 3:
        M, N = x.shape[1:]
        return x.repeat(n).reshape(M, N, n).transpose(2, 0, 1)
    raise ValueError("input must be 2D or 3D")


@nb.njit([nb_Idw(nb_Idw, nb.int64), nb_Rbf(nb_Rbf, nb.int64)], cache=True, nogil=True)
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
    if len(mdl) == 3:  # isinstance(mdl, Idw):
        return Idw(
            mdl.exp_weighting,
            np.ascontiguousarray(repeat_along_first_axis(mdl.Xm_, n)),
            np.ascontiguousarray(repeat_along_first_axis(mdl.ym_, n)),
        )
    else:
        return Rbf(
            mdl.kernel,  # type: ignore[union-attr]
            mdl.eps,  # type: ignore[union-attr]
            mdl.svd_tol,  # type: ignore[union-attr]
            mdl.exp_weighting,
            np.ascontiguousarray(repeat_along_first_axis(mdl.Xm_, n)),
            np.ascontiguousarray(repeat_along_first_axis(mdl.ym_, n)),
            np.ascontiguousarray(
                repeat_along_first_axis(mdl.coef_, n)  # type: ignore[union-attr]
            ),
            np.ascontiguousarray(
                repeat_along_first_axis(mdl.Minv_, n)  # type: ignore[union-attr]
            ),
        )
