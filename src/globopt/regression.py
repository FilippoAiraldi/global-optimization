"""
Implementation of Radial Basis Function and Inverse Distance Weighting regression
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


# NOTE: Batching
# --------
# Here is discussed the batching methods used by botorch and in our regression
# implementations, and how the two are interfaced.
# See also: https://botorch.org/docs/batching
#
# Conventions
# ------------
# botorch uses the convention `b0 x b1 x ... x q x d`, where
# * `b-i` is the number of batches of candidates to evaluate in parallel; the
#   minimization of the acquisition function consists in minimizing its sum over the
#   batches, and taking the best one at last
# * `q` is the number of candidates to consider jointly per batch; often, the best is
#   taken out of these per batch (in other words, at each optimization iteration, the
#   acquisition function minimization yields `q` next candidates to observe)
# * `d` is the dimension of the design space of each `q`-th candidate.
# Note that while there might be more than one batch dimension, usually we need just one
# in important methods.
# On the other side, the implementation of the regression models uses a slightly
# different convention, which is motivated by how pytorch treats and broadcasts linear
# algebra operations. In particular, it uses `p x m x d`, where
# * `p` is the number of parallel regression models (this is only useful in the
#   non-myopic, where we need to progress many different models in parallel. This is
#   akin to the `b` dimension in botorch)
# * `m` is the number of training points. In case of prediction, we prefer to denote
#   this same dimension as `n`.
# Note that the use of `p` parallel regressors is only useful in the non-myopic case,
# where we need to progress many different models in parallel, by exploring different
# acquisition points. Instead, in the myopic case, we expect `p = 1`.
#
# Interfacing
# -----------
# We make here the distinction between the myopic and non-myopic case.
# * myopic case:
#   * `IdwAcquisitionFunction`: in this acquisition function, `q = 1`. Moreover, the
#     regressor should be unique, i.e., `p = 1`, even though we do not explictly check.
#     This means that, in practice, the `b` dimension is botorch is automatically
#     swapped in second place and used as the `m` (usually, we use `n` for prediction
#     points, and `m` for training). This is done because batching the regressor usually
#     is numerically poorer than passing `b` points in `m` dimension.
#   * `qIdwAcquisitionFunction`: here, while `q > 1` is supported, in practice, in
#     our optimization loops we decide to only compute one candidate per iteration.
#     Instead, `b` is the number of batches of `q` points. The acquisition function
#     is minimized over the sum of its batches, and for each the best candidate out of
#     `q` is taken.
# * non-myopic case:
#   TODO: would `b x q x 1 x d` work for regressors as repeated as `b x q x m x d`?


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
    scaled_diff = V.sqrt().mul(train_Y.transpose(-1, -2) - Y)
    return torch.linalg.vector_norm(scaled_diff, dim=-1, keepdim=True)


@trace((torch.rand(5, 4, 3), torch.rand(5, 4, 1), torch.rand(5, 7, 3)))
def _idw_predict(
    train_X: Tensor, train_Y: Tensor, X: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mean and scale for IDW regression."""
    W = torch.cdist(X, train_X).square().clamp_min(DELTA).reciprocal()
    W_sum_recipr = W.sum(-1, keepdim=True).reciprocal()
    V = W.mul(W_sum_recipr)
    mean = V.matmul(train_Y)
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
    X: Tensor, Y: Tensor, eps: Tensor, svd_tol: Tensor
) -> tuple[Tensor, Tensor]:
    """Fits the RBF regression model to the training data."""
    _, M = _cdist_and_inverse_quadratic_kernel(X, X, eps)
    U, S, VT = torch.linalg.svd(M)
    S = S.where(S > svd_tol, torch.inf)
    Minv = VT.transpose(-2, -1).div(S.unsqueeze(-2)).matmul(U.transpose(-2, -1))
    coeffs = Minv.matmul(Y)
    return Minv, coeffs


@script(
    example_inputs=[
        (
            torch.rand(2, 5, 3),
            torch.rand(2, 5, 1),
            torch.rand(()),
            torch.rand(2, 3, 3),
            torch.rand(2, 3, 1),
        )
    ]
)  # unable to trace this one
def _rbf_partial_fit(
    X: Tensor, Y: Tensor, eps: Tensor, Minv: Tensor, coeffs: Tensor
) -> tuple[Tensor, Tensor]:
    """Fits the given RBF regression to the new training data."""
    n = coeffs.shape[-2]  # index of the first new data point onwards
    X_new = X[..., n:, :]
    _, Phi_and_phi = _cdist_and_inverse_quadratic_kernel(X_new, X, eps)
    PhiT = Phi_and_phi[..., :n]
    phi = Phi_and_phi[..., n:]
    L = Minv.matmul(PhiT.transpose(-2, -1))
    S = phi - PhiT.matmul(L)  # Schur complement
    Sinv = torch.linalg.inv(S)
    B = -L.matmul(Sinv)
    A = Minv - B.matmul(L.transpose(-2, -1))
    Minv_new = torch.cat(
        (torch.cat((A, B), -1), torch.cat((B.transpose(-2, -1), Sinv), -1)), -2
    )
    coeffs_new = Minv_new.matmul(Y)
    return Minv_new, coeffs_new


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
    # NOTE: here, we do not use `KernelLinearOperator` so as to avoid computing the
    # distance from `X` to `train_X` twice, one in the linear operator and one in the
    # `idw_weighting` function.
    dist, M = _cdist_and_inverse_quadratic_kernel(X, train_X, eps)
    mean = M.matmul(coeffs)
    W = dist.square().clamp_min(DELTA).reciprocal()
    W_sum_recipr = W.sum(-1, keepdim=True).reciprocal()
    V = W.mul(W_sum_recipr)
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
        return X  # do nothing


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
        return Idw(
            torch.cat((self.train_X, X), dim=-2), torch.cat((self.train_Y, Y), dim=-2)
        )


class Rbf(BaseRegression):
    """Radial Basis Function regression model in Global Optimization."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        eps: Union[float, Tensor] = 1.0,
        svd_tol: Union[float, Tensor] = 1e-8,
        Minv_and_coeffs: Optional[tuple[Tensor, Tensor]] = None,
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
        svd_tol : float, optional
            Tolerance for singular value decomposition for inversion, by default `1e-8`.
        Minv_and_coeffs : tuple of 2 Tensors, optional
            Precomputed inverse of the RBF kernel matrix and coefficients. By default
            `None`, in which case the model is fit anew to the training data. If
            provided, the model is only partially fit to the new data.
        """
        super().__init__(train_X, train_Y)
        eps = torch.scalar_tensor(eps)
        svd_tol = torch.scalar_tensor(svd_tol)
        if Minv_and_coeffs is None:
            Minv, coeffs = _rbf_fit(self.train_X, self.train_Y, eps, svd_tol)
        else:
            Minv, coeffs = _rbf_partial_fit(
                self.train_X, self.train_Y, eps, *Minv_and_coeffs
            )
        self.register_buffer("eps", eps)
        self.register_buffer("svd_tol", svd_tol)
        self.register_buffer("Minv", Minv)
        self.register_buffer("coeffs", coeffs)
        self.to(train_X)

    @property
    def Minv_and_coeffs(self) -> tuple[Tensor, Tensor]:
        """States of a fitted RBF regressor, i.e., the inverse of the kernel matrix and
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
        return _rbf_predict(self.train_X, self.train_Y, self.eps, self.coeffs, X)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **_: Any) -> "Rbf":
        return Rbf(
            torch.cat((self.train_X, X), dim=-2),
            torch.cat((self.train_Y, Y), dim=-2),
            self.eps,
            self.svd_tol,
            self.Minv_and_coeffs,
        )
