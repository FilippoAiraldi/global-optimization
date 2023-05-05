"""
Implementation of Radial Basis Function and Inverse Distance Weighting regression
according to [1]. These regression models are coded in line with the sklearn API, so
they offer common methods such as `fit`, `predict`, as well as `partial_fit`.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from numbers import Real
from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

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
        W *= np.exp(-W)
    return W


class IDWRegression(RegressorMixin, BaseEstimator):
    """Inverse Distance Weighting regression."""

    _parameter_constraints: dict[str, Any] = {"exp_weighting": ["boolean"]}

    def __init__(self, exp_weighting: bool = False) -> None:
        """Instantiate a regression model with IDWs.

        Parameters
        ----------
        exp_weighting : bool, optional
            Whether the weighting function should decay exponentially, by default
            `False`.
        """
        RegressorMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.exp_weighting = exp_weighting

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """Fits the model to the data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input data to be fitted.
        y : array_like of shape (n_samples,)
            The target values to be fitted.
        """
        self._validate_params()
        X, y = self._validate_data(X, y, y_numeric=True)
        self.X_ = X
        self.y_ = y
        return self

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """Partially fits the model to the data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input data to be fitted.
        y : array_like of shape (n_samples,)
            The target values to be fitted.
        """
        first_call = not hasattr(self, "X_")
        X, y = self._validate_data(X, y, y_numeric=True, reset=first_call)
        if first_call:
            self._validate_params()
            self.X_ = X
            self.y_ = y
        else:
            check_is_fitted(self, attributes=("X_", "y_"))
            self.X_ = np.concatenate((self.X_, X), axis=0)
            self.y_ = np.concatenate((self.y_, y), axis=0)
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Predicts target values.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input data for which `y` has to be predicted.

        Returns
        -------
        y: array of floats
            Prediction of `y`.
        """
        check_is_fitted(self, attributes=("X_", "y_"))
        X = self._validate_data(X, reset=False)

        # create matrix of inverse-distance weights
        W = idw_weighting(X, self.X_, self.exp_weighting)

        # predict as weighted average
        v = W / W.sum(axis=0)
        return v.T @ self.y_


"""RBF kernels."""
RBF_FUNCS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
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


class RBFRegression(RegressorMixin, BaseEstimator):
    """Radial Basis Function regression."""

    _parameter_constraints: dict[str, Any] = {
        "kernel": [StrOptions(RBF_FUNCS.keys())],
        "eps": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        kernel: Literal[
            "inversequadratic",
            "multiquadric",
            "linear",
            "gaussian",
            "thinplatespline",
            "inversemultiquadric",
        ] = "inversequadratic",
        eps: float = 1.0775,
    ) -> None:
        """Instantiate a regression model with RBFs.

        Parameters
        ----------
        kernel : {'inversequadratic', 'multiquadric', 'linear', 'gaussian',
            'thinplatespline', 'inversemultiquadric' }
            The type of RBF kernel to use.
        eps : float, optional
            Distance-scaling parameter for the RBF kernel, by default `1e-1`.
        """
        RegressorMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.kernel = kernel
        self.eps = eps

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """Fits the model to the data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input data to be fitted.
        y : array_like of shape (n_samples,)
            The target values to be fitted.
        """
        self._validate_params()
        X, y = self._validate_data(X, y, y_numeric=True)
        y = y.astype(float)  # type: ignore[union-attr]

        # create matrix of kernel evaluations
        fun = RBF_FUNCS[self.kernel]
        d2 = pdist(X, "sqeuclidean")  # returns all single distances, not a matrix
        M = squareform(fun(d2, self.eps))
        M[np.diag_indices_from(M)] = fun(0, self.eps)  # type: ignore[arg-type]

        # compute coefficients via SVD and inverse of M (useful for partial_fit)
        self.coef_, self.Minv_ = _linsolve_via_svd(M, y)

        # save training data
        self.X_ = X
        self.y_ = y
        return self

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """Partially fits the model to the data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input data to be fitted.
        y : array_like of shape (n_samples,)
            The target values to be fitted.
        """
        first_call = not hasattr(self, "X_")
        X, y = self._validate_data(X, y, y_numeric=True, reset=first_call)
        if first_call:
            return self.fit(X, y)

        check_is_fitted(self, attributes=("X_", "y_", "coef_", "Minv_"))
        y = y.astype(float)  # type: ignore[union-attr]

        # create matrix of kernel evaluations of new elements w.r.t. training data and
        # themselves
        fun = RBF_FUNCS[self.kernel]
        Phi = fun(cdist(self.X_, X, "sqeuclidean"), self.eps)
        phi = pdist(X, "sqeuclidean")  # returns all single distances, not a matrix
        phi = squareform(fun(phi, self.eps))
        phi[np.diag_indices_from(phi)] = fun(0, self.eps)  # type: ignore[arg-type]

        # update inverse of M via blockwise inversion and coefficients
        y_new, self.coef_, self.Minv_ = _blockwise_inversion(
            self.y_, y, self.Minv_, phi, Phi
        )

        # append to training data
        self.X_ = np.concatenate((self.X_, X), axis=0)
        self.y_ = y_new
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Predicts target values.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input data for which `y` has to be predicted.

        Returns
        -------
        y: array of floats
            Prediction of `y`.
        """
        check_is_fitted(self, attributes=("X_", "coef_"))
        X = self._validate_data(X, reset=False)

        # create matrix of kernel evaluations
        d2 = cdist(self.X_, X, "sqeuclidean")
        M = RBF_FUNCS[self.kernel](d2, self.eps)

        # predict as linear combination
        return M.T @ self.coef_
