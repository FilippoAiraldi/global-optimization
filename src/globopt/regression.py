"""
Implementation of Radial Basis Function and Inverse Distance Weighting regression
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from math import exp, sqrt
from typing import Any, Optional, Union

import numpy as np
import torch
from botorch.models.model import FantasizeMixin, Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from scipy.stats import lognorm
from sklearn.model_selection import KFold
from torch import Tensor
from torch.nn import Module

HALF_EXP_MINUS_2SQRT2 = exp(-2 * sqrt(2)) / 2
TWICESQRT3 = sqrt(3) * 2

DELTA = 1e-12
"""Small value to avoid division by zero."""


def _idw_scale(Y: Tensor, train_Y: Tensor, V: Tensor) -> Tensor:
    """Computes the IDW standard deviation function.

    Parameters
    ----------
    Y : Tensor
        `(b0 x b1 x ...) x n x 1` estimates of function values at the candidate points.
    train_Y : Tensor
        `(b0 x b1 x ...) x m x 1` training data points.
    V : Tensor
        `(b0 x b1 x ...) x n x m` normalized IDW weights.

    Returns
    -------
    Tensor
        The standard deviation of shape `(b0 x b1 x ...) x n x 1`.
    """
    scaled_diff = V.sqrt() * (train_Y.mT - Y)
    return torch.linalg.vector_norm(scaled_diff, dim=-1, keepdim=True)


def _idw_predict(
    train_X: Tensor, train_Y: Tensor, X: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mean and scale for IDW regression."""
    W = torch.cdist(X, train_X).square().clamp_min(DELTA).reciprocal()
    W_sum_recipr = W.sum(-1, keepdim=True).reciprocal()
    V = W * W_sum_recipr
    mean = V @ train_Y
    std = _idw_scale(mean, train_Y, V)
    return mean, std, W_sum_recipr, V


_idw_scale_jit = torch.jit.trace(
    _idw_scale, (torch.rand(5, 4, 1), torch.rand(5, 3, 1), torch.rand(5, 4, 3))
)
_idw_predict_jit = torch.jit.trace(
    _idw_predict, (torch.rand(5, 4, 3), torch.rand(5, 4, 1), torch.rand(5, 7, 3))
)


def _cdist_and_inverse_quadratic_kernel(
    X: Tensor, Xother: Tensor, eps: Tensor
) -> tuple[Tensor, Tensor]:
    """RBF inverse quadratic kernel function."""
    cdist = torch.cdist(X, Xother)
    scaled_cdist = eps[..., None, None] * cdist
    kernel = (scaled_cdist.square() + 1).reciprocal()
    return cdist, kernel


_cdist_and_inverse_quadratic_kernel_jit = torch.jit.trace(
    _cdist_and_inverse_quadratic_kernel,
    (torch.rand(5, 4, 3), torch.rand(5, 7, 3), torch.rand(())),
)


def _rbf_fit(
    X: Tensor, Y: Tensor, eps: Tensor, svd_tol: Tensor
) -> tuple[Tensor, Tensor]:
    """Fits the RBF regression model to the training data."""
    _, M = _cdist_and_inverse_quadratic_kernel(X, X, eps)
    U, S, VT = torch.linalg.svd(M)
    S = S.where(S >= svd_tol, torch.inf)
    Minv = (VT.mT / S.unsqueeze(-2)) @ U.mT
    coeffs = Minv.matmul(Y)
    return Minv, coeffs


def _rbf_partial_fit(
    X: Tensor, Y: Tensor, eps: Tensor, svd_tol: Tensor, Minv: Tensor, coeffs: Tensor
) -> tuple[Tensor, Tensor]:
    """Fits the given RBF regression to the new training data."""
    n = coeffs.shape[-2]  # index of the first new data point onwards
    X_new = X[..., n:, :]
    _, Phi_and_phi = _cdist_and_inverse_quadratic_kernel_jit(X_new, X, eps)
    PhiT = Phi_and_phi[..., :n]
    phi = Phi_and_phi[..., n:]

    # compute the new inverse of the kernel matrix via block matrix inversion and, where
    # it fails, from scratch
    Phi = PhiT.mT
    L = Minv @ Phi
    C = phi - PhiT @ L  # Schur complement
    C_det = torch.linalg.det(C)
    mask = torch.logical_or(C_det >= 1e-17, C_det <= -1e-17)  # magic number
    if mask.all().item():
        # all schur batches are invertible
        Cinv = torch.linalg.inv(C)
        B = -L @ Cinv
        A = Minv - B @ L.mT
        Minv_new = torch.cat((torch.cat((A, B), -1), torch.cat((B.mT, Cinv), -1)), -2)
    else:
        # some of the schur batches were not invertible. For those that were invertible,
        # use block matrix inversion. For those that were not, compute the full inverse
        Minv_new_shape = X.shape[:-2] + (X.shape[-2], X.shape[-2])
        Minv_new = torch.empty(Minv_new_shape, device=Minv.device, dtype=Minv.dtype)

        # block inversion
        Cinv = torch.linalg.inv(C[mask, :, :])
        L = L[mask, :, :]
        B = -L @ Cinv
        if Minv.ndim < Minv_new.ndim:
            Minv = Minv.expand(X.shape[:-2] + (n, n))
        A = Minv[mask, :, :] - B @ L.mT
        Minv_new[mask, :, :] = torch.cat(
            (torch.cat((A, B), -1), torch.cat((B.mT, Cinv), -1)), -2
        )

        # full inversion
        not_mask = ~mask
        X_old_nm = X[..., :n, :][not_mask, :, :]
        _, M = _cdist_and_inverse_quadratic_kernel_jit(X_old_nm, X_old_nm, eps)
        M_new = torch.cat(
            (torch.cat((M, Phi[not_mask, :, :]), -1), Phi_and_phi[not_mask, :, :]), -2
        )
        U_new, S_new, VT_new = torch.linalg.svd(M_new)
        S_new = S_new.where(S_new >= svd_tol, torch.inf)
        Minv_new[not_mask, :, :] = (VT_new.mT / S_new.unsqueeze(-2)) @ U_new.mT

    # finally, compute the new coefficients
    coeffs_new = Minv_new @ Y
    return Minv_new, coeffs_new


def _rbf_predict(
    train_X: Tensor, train_Y: Tensor, eps: Tensor, coeffs: Tensor, X: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Predicts mean and scale for RBF regression."""
    dist, M = _cdist_and_inverse_quadratic_kernel(X, train_X, eps)
    mean = M @ coeffs
    W = dist.square().clamp_min(DELTA).reciprocal()
    W_sum_recipr = W.sum(-1, keepdim=True).reciprocal()
    V = W * W_sum_recipr
    std = _idw_scale(mean, train_Y, V)
    return mean, std, W_sum_recipr, V


_rbf_fit_jit = torch.jit.trace(
    _rbf_fit, (torch.rand(5, 4, 3), torch.rand(5, 4, 1), torch.rand(()), torch.rand(()))
)
_rbf_partial_fit_jit = torch.jit.script(
    _rbf_partial_fit,
    example_inputs=[
        (
            torch.rand(5, 4, 3),
            torch.rand(5, 4, 1),
            torch.rand(()),
            torch.rand(()),
            torch.rand(5, 2, 2),
            torch.rand(5, 2, 1),
        )
    ],
)
_rbf_predict_jit = torch.jit.trace(
    _rbf_predict,
    (
        torch.rand(5, 4, 3),
        torch.rand(5, 4, 1),
        torch.rand(()),
        torch.rand(5, 4, 1),
        torch.rand(5, 3, 3),
    ),
)


class BaseRegression(Model, FantasizeMixin):
    """Base class for a regression model."""

    def __init__(self, train_X: Tensor, train_Y: Tensor) -> None:
        """Instantiates a regression model for Global Optimization.

        Parameters
        ----------
        train_X : Tensor
            A `(b0 x b1 x ...) x m x d`, where `m` is the number of training points,
            `d` is the dimension of each point, and `b`s are the size of
            batched/parallel regressors to train.
        train_Y : Tensor
            A `(b0 x b1 x ...) x m x 1` tensor of evaluation corresponding to the
            `train_X` points.
        """
        Model.__init__(self)
        FantasizeMixin.__init__(self)
        self.train_X = train_X
        self.train_Y = train_Y.unsqueeze(-1) if train_Y.ndim == 1 else train_Y
        self.to(train_X)  # make sure we're on the right device/dtype

    @property
    def num_outputs(self) -> int:
        return 1  # only one output is supported

    @property
    def likelihood(self) -> None:
        """No likelihood is supported (just for compatibility with `FantasizeMixin`)."""
        return None

    def posterior(self, X: Tensor, **_: Any) -> GPyTorchPosterior:
        self.eval()
        # NOTE: do not modify input/output shapes here. It is the responsibility of the
        # acquisition function calling this method to do so.
        mean, scale, W_sum_recipr, V = self.forward(X)
        # NOTE: it's a bit sketchy, but `W_sum_recipr` is needed by the acquisition
        # functions. It gets first computed here, so it is convenient to manually attach
        # it to the model for later re-use.
        distribution = MultivariateNormal(
            mean.squeeze(-1), DiagLinearOperator(scale.square().squeeze(-1))
        )
        posterior = GPyTorchPosterior(distribution)
        posterior._scale = scale
        posterior._W_sum_recipr = W_sum_recipr
        posterior._V = V
        return posterior

    def transform_inputs(self, X: Tensor, _: Optional[Module] = None) -> Tensor:
        return X  # does nothing

    def _prepare_for_fantasizing(self, X: Tensor, Y: Tensor) -> tuple[Tensor, Tensor]:
        """Prepares and shapes the training data for fantasizing."""
        # X:           num_fantasies... x batch_shape x n' x m
        # Y: samples x num_fantasies... x batch_shape x n' x 1
        train_X = self.train_X
        dim_diff = X.ndim - train_X.ndim
        if dim_diff:
            train_X = train_X.view(*([1] * dim_diff), *train_X.shape)
            train_X = train_X.expand(*X.shape[:-2], -1, -1)
        train_Y = self.train_Y
        dim_diff = Y.ndim - train_Y.ndim
        if dim_diff:
            train_Y = train_Y.view(*([1] * dim_diff), *train_Y.shape)
            train_Y = train_Y.expand(*Y.shape[:-2], -1, -1)
        return train_X, train_Y


class Idw(BaseRegression):
    """Inverse Distance Weighting regression model in Global Optimization."""

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Computes the IDW regression model.

        Parameters
        ----------
        X : Tensor
            A `(b0 x b1 x ...) x n x 1` tensor of `d`-dim design points, where `n` is
            the number of candidate points to estimate, and `b`s are the batched
            regressor sizes.

        Returns
        -------
        tuple of 3 Tensors
            Returns
                - the mean estimate `(b0 x b1 x ...) x n x 1`
                - the standard deviation of the estimate `(b0 x b1 x ...) x n x 1`
                - the reciprocal of the sum of the IDW weights `(b0 x b1 x ...) x n x 1`
                - the normalized IDW weights `(b0 x b1 x ...) x n x m`, where `m` are
                  the number of training points.
        """
        return _idw_predict_jit(self.train_X, self.train_Y, X)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **_: Any) -> "Idw":
        train_X, train_Y = self._prepare_for_fantasizing(X, Y)
        return Idw(torch.cat((train_X, X), dim=-2), torch.cat((train_Y, Y), dim=-2))

    def __str__(self) -> str:
        return self.__class__.__name__


class Rbf(BaseRegression):
    """Radial Basis Function regression model in Global Optimization."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        eps: Union[None, float, Tensor] = None,
        svd_tol: Union[float, Tensor] = 1e-8,
        init_state: Optional[tuple[Tensor, Tensor]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Instantiates an RBF regression model for Global Optimization.

        Parameters
        ----------
        train_X : Tensor
            A `(b0 x b1 x ...) x m x d`, where `m` is the number of training points,
            `d` is the dimension of each point, and `b`s are the size of
            batched/parallel regressors to train.
        train_Y : Tensor
            A `(b0 x b1 x ...) x m x 1` tensor of evaluation corresponding to the
            `train_X` points.
        eps : float, optional
            Distance-scaling parameter for the RBF kernel. If `None`, its value is
            automatically determined via cross-validation. By default `None`.
        svd_tol : float, optional
            Tolerance for singular value decomposition for inversion, by default `1e-8`.
        init_state : tuple of 3 Tensors, optional
            Initial state of the regressor, in case of previous partial fitting, made up
            of the inverse of previous kernel distance matrix and the RBF coefficients.
            This is a tuple of
                - `Minv (b0 x b1 x ...) x m' x m'`
                - `coeffs (b0 x b1 x ...) x m' x 1`
            where `m'` are the number of training points in the previous fitting.
            By default `None`, in which case the model is fit anew to the training data.
            This initial state is also disregarded if `eps` is `None`.
        rng : int or RandomState, optional
            Random state for the cross-validation, by default `None`.

        Raises
        ------
        ValueError
            If `eps` is `None` and the model is not fit to a single batch of data, i.e.,
            `train_X` has more than `2` dimensions.
        """
        super().__init__(train_X, train_Y)
        svd_tol = torch.scalar_tensor(svd_tol)
        use_cv = eps is None
        if use_cv:
            # when fantasizing, avoid performing cross-validation
            if self.train_X.ndim != 2:
                raise ValueError(
                    "Cannot fit `eps` via cross-validation when fitting a model with "
                    "multiple batches."
                )
            eps = self.eps_cross_validation(
                self.train_X, self.train_Y, svd_tol, rng=rng
            )
        eps = torch.scalar_tensor(eps)
        if use_cv or init_state is None:
            # if eps was tuned via CV, we must fit the model from scratch as the
            # previous state was most likely fitted with a different eps
            Minv, coeffs = _rbf_fit_jit(self.train_X, self.train_Y, eps, svd_tol)
        else:
            Minv, coeffs = _rbf_partial_fit_jit(
                self.train_X, self.train_Y, eps, svd_tol, *init_state
            )
        self.register_buffer("eps", eps)
        self.register_buffer("svd_tol", svd_tol)
        self.register_buffer("Minv", Minv)
        self.register_buffer("coeffs", coeffs)
        self.to(train_X)

    @property
    def state(self) -> tuple[Tensor, Tensor]:
        """State of a fitted RBF regressor, i.e., the inverse of the kernel matrix and
        coefficients. Use this to partially fit a new regressor (see `__init__`)"""
        return self.Minv, self.coeffs

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Computes the RBF regression model.

        Parameters
        ----------
        X : Tensor
            A `(b0 x b1 x ...) x n x d` tensor of `d`-dim design points, where `n` is
            the number of candidate points to estimate, and `b`s are the batched
            regressor sizes.

        Returns
        -------
        tuple of 3 Tensors
            Returns
                - the mean estimate `(b0 x b1 x ...) x n x 1`
                - the standard deviation of the estimate `(b0 x b1 x ...) x n x 1`
                - the reciprocal of the sum of the IDW weights `(b0 x b1 x ...) x n x 1`
                - the normalized IDW weights `(b0 x b1 x ...) x n x m`, where `m` are
                  the number of training points.
        """
        return _rbf_predict_jit(self.train_X, self.train_Y, self.eps, self.coeffs, X)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **_: Any) -> "Rbf":
        train_X, train_Y = self._prepare_for_fantasizing(X, Y)
        Xnew = torch.cat((train_X, X), dim=-2)
        Ynew = torch.cat((train_Y, Y), dim=-2)
        return Rbf(Xnew, Ynew, self.eps, self.svd_tol, self.state)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps})"

    @staticmethod
    def eps_cross_validation(
        X: Tensor,
        Y: Tensor,
        svd_tol: Tensor,
        n_iter: int = 20,
        max_n_splits: int = 5,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Computes the optimal `eps` parameter for the RBF kernel via cross-validation.
        This is done by sampling `eps` from a log-normal distribution and selecting the
        one that minimizes the prediction error on a test set.

        Parameters
        ----------
        X : Tensor
            The training data points of shape `m x d`, where `m` is the number of
            training points and `d` is the dimension of each point.
        Y : Tensor
            The training data values of shape `m x 1`.
        svd_tol : Tensor
            Tolerance for singular value decomposition for inversion.
        n_iter : int, optional
            Number of iterations for the random cross-validation, by default `20`.
        max_n_splits : int, optional
            Maximum number of splits for the cross-validation, by default `5`.
        rng : np.random.Generator, optional
            Random state for the cross-validation, by default `None`.

        Returns
        -------
        float
            The optimal `eps` parameter for the RBF kernel.
        """
        n_samples, dim = X.shape
        n_splits = min(max_n_splits, n_samples)

        eps_dist = lognorm(s=TWICESQRT3, scale=HALF_EXP_MINUS_2SQRT2 / dim, loc=1e-3)
        eps_samples = torch.as_tensor(
            eps_dist.rvs(size=n_iter, random_state=rng), dtype=X.dtype, device=X.device
        )

        splitter = KFold(n_splits=n_splits)
        if n_samples % n_splits != 0:
            # we cannot vectorize cross-validation computations, so we use a for loop
            scores = torch.zeros((n_iter,), dtype=X.dtype, device=X.device)
            for train_idx, test_idx in splitter.split(X, Y):
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                X_test = X[test_idx]
                Y_test = Y[test_idx]
                _, coeffs = _rbf_fit(X_train, Y_train, eps_samples, svd_tol)
                Y_pred, _, _, _ = _rbf_predict(
                    X_train, Y_train, eps_samples, coeffs, X_test
                )
                scores += (Y_test - Y_pred).square().sum(dim=(-1, -2))
        else:
            # we can vectorize here instead
            train_idx_, test_idx_ = zip(*splitter.split(X, Y))
            train_idx = np.asarray(train_idx_)
            test_idx = np.asarray(test_idx_)
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]
            eps_samples_ = eps_samples.unsqueeze(-1)
            _, coeffs = _rbf_fit(X_train, Y_train, eps_samples_, svd_tol)
            Y_pred, _, _, _ = _rbf_predict(
                X_train, Y_train, eps_samples_, coeffs, X_test
            )
            scores = (Y_test - Y_pred).square().sum(dim=(-1, -2, -3))

        return eps_samples[scores.argmin()].item()
