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
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from globopt.core.functional_regression import (
    RBF_FUNCS,
    RbfKernel,
    idw_fit,
    idw_partial_fit,
    idw_predict,
    rbf_fit,
    rbf_partial_fit,
    rbf_predict,
)


class IdwRegression(RegressorMixin, BaseEstimator):
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
        self.X_, self.y_ = idw_fit(X, y)  # type: ignore[arg-type]
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
        if not hasattr(self, "X_"):
            return self.fit(X, y)

        X, y = self._validate_data(X, y, y_numeric=True, reset=False)
        check_is_fitted(self, attributes=("X_", "y_"))
        self.X_, self.y_ = idw_partial_fit((self.X_, self.y_), X, y)
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
        return idw_predict(
            (self.X_, self.y_), X, self.exp_weighting  # type: ignore[arg-type]
        )


class RbfRegression(RegressorMixin, BaseEstimator):
    """Radial Basis Function regression."""

    _parameter_constraints: dict[str, Any] = {
        "kernel": [StrOptions(RBF_FUNCS.keys())],  # can we add RbfKernel?
        "eps": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        kernel: RbfKernel = "inversequadratic",
        eps: float = 1.0775,
    ) -> None:
        """Instantiate a regression model with RBFs.

        Parameters
        ----------
        kernel : {'inversequadratic', 'multiquadric', 'linear', 'gaussian',
            'thinplatespline', 'inversemultiquadric' }
            The type of RBF kernel to use.
        eps : float, optional
            Distance-scaling parameter for the RBF kernel, by default `1.0775`.
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
        self.X_, self.y_, self.coef_, self.Minv_ = rbf_fit(
            X, y, self.kernel, self.eps  # type: ignore[arg-type]
        )
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
        if not hasattr(self, "X_"):
            return self.fit(X, y)

        X, y = self._validate_data(X, y, y_numeric=True, reset=False)
        check_is_fitted(self, attributes=("X_", "y_", "coef_", "Minv_"))
        y = y.astype(float)  # type: ignore[union-attr]
        fitresult = (self.X_, self.y_, self.coef_, self.Minv_)
        self.X_, self.y_, self.coef_, self.Minv_ = rbf_partial_fit(
            fitresult, X, y, self.kernel, self.eps  # type: ignore[arg-type]
        )
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
        fitresult = (self.X_, self.y_, self.coef_, self.Minv_)
        return rbf_predict(
            fitresult, X, self.kernel, self.eps  # type: ignore[arg-type]
        )
