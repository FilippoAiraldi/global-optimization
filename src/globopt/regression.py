"""
Implementation of Radial Basis Function and Inverse Distance Weighting regression
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Any, Optional, Union

import torch
from botorch.models.model import FantasizeMixin, Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from torch import Tensor
from torch.nn import Module

DELTA = 1e-12
"""Small value to avoid division by zero."""


def trace(*args: Any, **kwargs: Any):
    """Applies `torch.jit.trace` to the decorated function."""

    def _inner_decorator(func):
        return torch.jit.trace(func, *args, **kwargs)

    return _inner_decorator


def script(*args: Any, **kwargs: Any):
    """Applies `torch.jit.script` to the decorated function."""

    def _inner_decorator(func):
        return torch.jit.script(func, *args, **kwargs)

    return _inner_decorator


@trace((torch.rand(5, 4, 1), torch.rand(5, 3, 1), torch.rand(5, 4, 3)))
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


@trace((torch.rand(5, 4, 3), torch.rand(5, 4, 1), torch.rand(5, 7, 3)))
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


@trace((torch.rand(5, 4, 3), torch.rand(5, 7, 3), torch.rand(())))
def _cdist_and_inverse_quadratic_kernel(
    X: Tensor, Xother: Tensor, eps: Tensor
) -> tuple[Tensor, Tensor]:
    """RBF inverse quadratic kernel function."""
    cdist = torch.cdist(X, Xother)
    scaled_cdist = eps[..., None, None] * cdist
    kernel = (scaled_cdist.square() + 1).reciprocal()
    return cdist, kernel


@trace((torch.rand(5, 4, 3), torch.rand(5, 4, 1), torch.rand(()), torch.rand(())))
def _rbf_fit(
    X: Tensor, Y: Tensor, eps: Tensor, eig_tol: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Fits the RBF regression model to the training data."""
    _, M = _cdist_and_inverse_quadratic_kernel(X, X, eps)
    eigvals, eigvecs = torch.linalg.eigh(M)
    eigvals_thresholded = eigvals.where(eigvals.abs() > eig_tol, torch.inf)
    Minv = (eigvecs / eigvals_thresholded.unsqueeze(-2)) @ eigvecs.mT
    coeffs = Minv.matmul(Y)
    return eigvals, eigvecs, coeffs


@script(  # unable to trace this one
    example_inputs=[
        (
            torch.rand(5, 4, 3),
            torch.rand(5, 4, 1),
            torch.rand(()),
            torch.rand(()),
            torch.rand(5, 2),
            torch.rand(5, 2, 2),
            torch.rand(5, 2, 1),
        )
    ]
)
def _rbf_partial_fit(
    X: Tensor,
    Y: Tensor,
    eps: Tensor,
    eig_tol: Tensor,
    eigvals: Tensor,
    eigvecs: Tensor,
    coeffs: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fits the given RBF regression to the new training data."""
    n = coeffs.shape[-2]  # index of the first new data point onwards
    X_new = X[..., n:, :]
    _, Phi_and_phi = _cdist_and_inverse_quadratic_kernel(X_new, X, eps)
    PhiT = Phi_and_phi[..., :n]
    phi = Phi_and_phi[..., n:]
    prod = PhiT @ eigvecs
    eigvals_mat = eigvals.expand(prod.mT.shape[:-1]).diag_embed()
    M_proxy_new = torch.cat(
        (torch.cat((eigvals_mat, prod.mT), -1), torch.cat((prod, phi), -1)), -2
    )
    eigvals_new, eigvecs_tmp = torch.linalg.eigh(M_proxy_new)
    eigvecs_new = torch.cat(
        (eigvecs @ eigvecs_tmp[..., :n, :], eigvecs_tmp[..., n:, :]), -2
    )
    eigvals_thresholded = eigvals_new.where(eigvals_new.abs() > eig_tol, torch.inf)
    Minv_new = (eigvecs_new / eigvals_thresholded.unsqueeze(-2)) @ eigvecs_new.mT
    coeffs_new = Minv_new @ Y
    return eigvals_new, eigvecs_new, coeffs_new


@trace(
    (
        torch.rand(5, 4, 3),
        torch.rand(5, 4, 1),
        torch.rand(()),
        torch.rand(5, 4, 1),
        torch.rand(5, 3, 3),
    )
)
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
        return _idw_predict(self.train_X, self.train_Y, X)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **_: Any) -> "Idw":
        train_X, train_Y = self._prepare_for_fantasizing(X, Y)
        return Idw(torch.cat((train_X, X), dim=-2), torch.cat((train_Y, Y), dim=-2))


class Rbf(BaseRegression):
    """Radial Basis Function regression model in Global Optimization."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        eps: Union[float, Tensor] = 1.0,
        eig_tol: Union[float, Tensor] = 1e-8,
        init_state: Optional[tuple[Tensor, Tensor]] = None,
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
            Distance-scaling parameter for the RBF kernel, by default `1.0`.
        eig_tol : float, optional
            Tolerance for singular value decomposition for inversion, by default `1e-8`.
        init_state : tuple of 3 Tensors, optional
            Initial state of the regressor, in case of previous partial fitting, made up
            of the eigendecomposition of the previous kernel distance matrix. This is a
            tuple of
                - `vals (b0 x b1 x ...) x m'`: the eigenvalues of the kernel matrix
                - `vec (b0 x b1 x ...) x m' x m'`: the eigenvectors of the kernel matrix
            where `m'` are the number of training points in the previous fitting.
            By default `None`, in which case the model is fit anew to the training data.
        """
        super().__init__(train_X, train_Y)
        eps = torch.scalar_tensor(eps)
        eig_tol = torch.scalar_tensor(eig_tol)
        if init_state is None:
            eigvals, eigvecs, coeffs = _rbf_fit(
                self.train_X, self.train_Y, eps, eig_tol
            )
        else:
            eigvals, eigvecs, coeffs = _rbf_partial_fit(
                self.train_X, self.train_Y, eps, eig_tol, *init_state
            )
        self.register_buffer("eps", eps)
        self.register_buffer("eig_tol", eig_tol)
        self.register_buffer("eigvals", eigvals)
        self.register_buffer("eigvecs", eigvecs)
        self.register_buffer("coeffs", coeffs)
        self.to(train_X)

    @property
    def state(self) -> tuple[Tensor, Tensor, Tensor]:
        """State of a fitted RBF regressor, i.e., the products of the eigendecomposition
        of the kernel matrix and coefficients. Use this to partially fit a new regressor
        (see `__init__`)"""
        return self.eigvals, self.eigvecs, self.coeffs

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
        return _rbf_predict(self.train_X, self.train_Y, self.eps, self.coeffs, X)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **_: Any) -> "Rbf":
        train_X, train_Y = self._prepare_for_fantasizing(X, Y)
        Xnew = torch.cat((train_X, X), dim=-2)
        Ynew = torch.cat((train_Y, Y), dim=-2)
        return Rbf(Xnew, Ynew, self.eps, self.eig_tol, self.state)
