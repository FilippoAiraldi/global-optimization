"""Implementation of Radial Basis Function and Inverse Distance Weighting regression."""


from typing import Any, Dict
from typing_extensions import Self
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import cdist


DELTA = 1e-9


class IDWRegression(RegressorMixin, BaseEstimator):
    """Inverse Distance Weighting regression."""

    _parameter_constraints: Dict[str, Any] = {"exp_weighting": ["boolean"]}

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
        first_call = not hasattr(self, "X_") or not hasattr(self, "y_")
        if first_call:
            self._validate_params()
        X, y = self._validate_data(X, y, y_numeric=True, reset=first_call)

        if first_call:
            self.X_ = X
            self.y_ = y
        else:
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
        d2 = cdist(self.X_, X, "sqeuclidean")
        W = 1 / (d2 + DELTA)
        if self.exp_weighting:
            W *= np.exp(-W)

        # predict as weighted average
        v = W / W.sum(axis=0)
        return v.T @ self.y_
