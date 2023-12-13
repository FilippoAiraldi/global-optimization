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
# * `b-i` is the number of batches of candidates to evaluate in parallel
# * `q` is the number of candidates to consider jointly
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
#   this same dimension as `n`.`
# Note that the use of `p` parallel regressors is only useful in the non-myopic case,
# where we need to progress many different models in parallel, by exploring different
# acquisition points. Instead, in the myopic case, we expect `p = 1`.
#
# Interfacing
# -----------
# We make here the distinction between the myopic and non-myopic case.
# * myopic case: for the simplest cases, i.e., the analytical acquisition functions, we
#   expect `q = 1`. This means that the `b` dimension is botorch can be swapped in
#   second place and used as the `m`. Usually, we call it `n` to distinguish the `m`
#   training points from the `n` prediction points.
#   For the Monte Carlo myopic case, TODO
# * non-myopic case: TODO: would `b x q x 1 x d` work for regressors as repeated as
# `b x q x m x d`?


from typing import Any, Optional, Union

import torch
from botorch.models.model import Model
from botorch.posteriors import TorchPosterior
from linear_operator import LinearOperator, to_linear_operator
from linear_operator.operators import KernelLinearOperator
from torch import Tensor
from torch.distributions import Normal

"""Small value to avoid division by zero."""
DELTA = torch.scalar_tensor(1e-8)


class BaseRegression(Model):
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
        super().__init__()
        self.train_X = train_X
        self.train_Y = train_Y
        self.to(train_X)  # make sure we're on the right device/dtype

    @property
    def num_outputs(self) -> int:
        return 1  # only one output is supported

    def posterior(self, X: Tensor, **_: Any) -> TorchPosterior:
        self.eval()
        # NOTE: do not modify input/output shapes here. It is the responsibility of the
        # acquisition function calling this method to do so.
        mean, scale, W_sum_recipr = self.forward(X)
        # NOTE: it's a bit sketchy, but `W_sum_recipr` is needed by the acquisition
        # functions. It gets first computed here, so it is convenient to manually attach
        # it to the model for later re-use.
        self.W_sum_recipr = W_sum_recipr
        return TorchPosterior(Normal(mean, scale))


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


def _idw_regression_predict(
    train_X: Tensor, train_Y: Tensor, X: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Mean and scale for IDW regression."""
    W = torch.cdist(X, train_X).square().clamp_min(DELTA).reciprocal()
    W_sum_recipr = W.sum(-1, keepdim=True).reciprocal()
    V = W.mul(W_sum_recipr)
    mean = V.matmul(train_Y)
    std = _idw_scale(mean, train_Y, V)
    return mean, std, W_sum_recipr


class Idw(BaseRegression):
    """Inverse Distance Weighting regression model in Global Optimization."""

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
                - the mean estimate
                - the idw standard deviation of the estimate
                - the reciprocal of the sum of the IDW weights
        """
        return _idw_regression_predict(self.train_X, self.train_Y, X)


def _cdist_and_inverse_quadratic_kernel(
    X: Tensor, Y: Tensor, eps: Tensor
) -> tuple[Tensor, Tensor]:
    """RBF inverse quadratic kernel function."""
    cdist = torch.cdist(X, Y)
    scaled_cdist = eps[..., None, None] * cdist
    kernel = torch.scalar_tensor(1.0).addcmul(scaled_cdist, scaled_cdist).reciprocal()
    return cdist, kernel


def _inverse_quadratic_kernel(X: Tensor, Y: Tensor, eps: Tensor) -> Tensor:
    """RBF inverse quadratic kernel function."""
    return _cdist_and_inverse_quadratic_kernel(X, Y, eps)[1]


def _rbf_fit(
    X: Tensor, Y: Tensor, eps: Tensor, svd_tol: Tensor
) -> tuple[LinearOperator, Tensor]:
    """Fits the RBF regression model to the training data."""
    M = KernelLinearOperator(
        X, X, _inverse_quadratic_kernel, eps=eps, num_nonbatch_dimensions={"eps": 0}
    )
    U, S, VT = torch.linalg.svd(M)
    S = S.where(S > svd_tol, torch.inf)
    Minv = VT.transpose(-2, -1).div(S.unsqueeze(-2)).matmul(U.transpose(-2, -1))
    coeffs = Minv.matmul(Y)
    return Minv, coeffs


def _rbf_partial_fit(
    X: Tensor,
    Y: Tensor,
    eps: Tensor,
    Minv: LinearOperator,
    coeffs: Tensor,
) -> tuple[LinearOperator, Tensor]:
    """Fits the given RBF regression to the new training data."""
    n = coeffs.size(-2)  # index of the first new data point onwards
    X_new = X[..., n:, :]
    Phi_and_phi = KernelLinearOperator(
        X_new, X, _inverse_quadratic_kernel, eps=eps, num_nonbatch_dimensions={"eps": 0}
    )
    PhiT = Phi_and_phi[..., :n]
    phi = Phi_and_phi[..., n:]
    L = Minv.matmul(PhiT.transpose(-2, -1))
    c = (phi - PhiT.matmul(L)).to_dense()  # TODO: force invertibility
    c_inv = torch.linalg.inv(c)
    B = -L.matmul(c_inv)
    A = (Minv - B.matmul(L.transpose(-2, -1))).to_dense()
    Minv_new = to_linear_operator(
        torch.cat(
            (torch.cat((A, B), -1), torch.cat((B.transpose(-2, -1), c_inv), -1)), -2
        )
    )
    coeffs_new = Minv_new.matmul(Y)
    return Minv_new, coeffs_new


def _rbf_predict(
    train_X: Tensor, train_Y: Tensor, eps: Tensor, coeffs: Tensor, X: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
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
    return mean, std, W_sum_recipr


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
        eps = torch.scalar_tensor(eps)
        svd_tol = torch.scalar_tensor(svd_tol)
        if Minv_and_coeffs is None:
            Minv, coeffs = _rbf_fit(train_X, train_Y, eps, svd_tol)
        else:
            Minv, coeffs = _rbf_partial_fit(train_X, train_Y, eps, *Minv_and_coeffs)
        super().__init__(train_X, train_Y)
        self.register_buffer("eps", eps)
        self.register_buffer("svd_tol", svd_tol)
        self.Minv = Minv  # cannot do `self.register_buffer("Minv", Minv)`
        self.register_buffer("coeffs", coeffs)

    def forward(self, X: Tensor) -> Normal:
        """Computes the RBF regression model.

        Parameters
        ----------
        X : Tensor
            A `(b0 x b1 x ...) x n x 1` tensor of `d`-dim design points, where `n` is
            the number of candidate points to estimate, and `b`s are the batched
            regressor sizes.

        Returns
        -------
        Normal
            The normal
        """
        return _rbf_predict(self.train_X, self.train_Y, self.eps, self.coeffs, X)
