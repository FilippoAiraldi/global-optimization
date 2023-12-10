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
from botorch.models.model import Model
from botorch.posteriors import TorchPosterior
from linear_operator import LinearOperator, to_linear_operator
from linear_operator.operators import KernelLinearOperator
from torch import Tensor
from torch.distributions import Normal

"""Small value to avoid division by zero."""
DELTA = torch.scalar_tensor(1e-8)


def _idw_scale(Y: Tensor, train_Y: Tensor, V: Tensor) -> Tensor:
    """Computes the IDW standard deviation function.

    Parameters
    ----------
    Y : Tensor
        `b x q x 1` estimates of the function values at the candidate points.
    train_Y : Tensor
        `b x m x 1` training data points.
    V : Tensor
        `b x q x m` normalized IDW weights.

    Returns
    -------
    Tensor
        The standard deviation of shape `b x q x 1`.
    """
    scaled_diff = V.sqrt().mul(train_Y.transpose(2, 1) - Y)
    return torch.linalg.vector_norm(scaled_diff, dim=2, keepdim=True)


trace_idw_scale = torch.jit.trace(
    _idw_scale, (torch.rand(7, 5, 1), torch.rand(7, 4, 1), torch.rand(7, 5, 4))
)


def _idw_regression_mean_and_std(
    train_X: Tensor, train_Y: Tensor, X: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Mean and scale for IDW regression."""
    W = torch.maximum(torch.cdist(X, train_X).square(), DELTA).reciprocal()
    W_sum_recipr = W.sum(2, keepdim=True).reciprocal()
    V = W.mul(W_sum_recipr)
    mean = V.bmm(train_Y)
    std = trace_idw_scale(mean, train_Y, V)
    return mean, std, W_sum_recipr


trace_idw_regression_mean_and_std = torch.jit.trace(
    _idw_regression_mean_and_std,
    (torch.rand(10, 5, 2), torch.rand(10, 5, 1), torch.rand(10, 2, 2)),
)


class BaseRegression(Model):
    """Base class for a regression model."""

    @property
    def num_outputs(self) -> int:
        return 1  # only one output is supported

    def posterior(self, X: Tensor, **_: Any) -> TorchPosterior:
        self.eval()
        mean, scale, _ = self.forward(X)
        return TorchPosterior(Normal(mean, scale))


class Idw(BaseRegression):
    """Inverse Distance Weighting regression model in Global Optimization."""

    def __init__(self, train_X: Tensor, train_Y: Tensor) -> None:
        """Instantiates an IDW regression model in Global Optimization.

        Parameters
        ----------
        train_X : Tensor
            A `b x m x d`-dim batched tensor of `m` training points with dimension `d`.
        train_Y : Tensor
            A `b x m x 1`-dim batched tensor of `m` training observations.
        """
        super().__init__()
        self.train_X = train_X
        self.train_Y = train_Y

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the IDW regression model.

        Parameters
        ----------
        X : Tensor
            A `b x q x d`-dim batched tensor of `d`-dim design points, where `q` is the
            number of candidate points to estimate, and `b` is the batched regressor
            size.

        Returns
        -------
        tuple of 3 Tensors
            Returns
                - the mean estimate
                - the idw standard deviation of the estimate
                - the reciprocal of the sum of the IDW weights
        """
        return trace_idw_regression_mean_and_std(self.train_X, self.train_Y, X)


def _inverse_quadratic_kernel(X: Tensor, Y: Tensor, eps: Tensor) -> Tensor:
    """RBF inverse quadratic kernel function."""
    scaled_dist = eps[..., None, None] * torch.cdist(X, Y)
    return torch.scalar_tensor(1.0).addcmul(scaled_dist, scaled_dist).reciprocal()


trace_invquad_kernel = torch.jit.trace(
    _inverse_quadratic_kernel,
    (torch.rand(10, 5, 2), torch.rand(10, 7, 2), torch.rand(10)),
)


def _rbf_fit(
    X: Tensor, Y: Tensor, eps: Tensor, svd_tol: Tensor
) -> tuple[LinearOperator, Tensor]:
    """Fits the RBF regression model to the training data."""
    # NOTE: here, we do not use jit because of `KernelLinearOperator`, but it seems
    # better than standard svd.
    M = KernelLinearOperator(
        X, X, trace_invquad_kernel, eps=eps, num_nonbatch_dimensions={"eps": 0}
    )
    U, S, VT = torch.linalg.svd(M)
    S = S.where(S > svd_tol, torch.inf)
    Minv = VT.transpose(1, 2).div(S.unsqueeze(1)).matmul(U.transpose(1, 2))
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
    n = coeffs.size(1)  # index of the first new data point onwards
    X_new = X[:, n:]
    Phi_and_phi = KernelLinearOperator(
        X_new, X, trace_invquad_kernel, eps=eps, num_nonbatch_dimensions={"eps": 0}
    )
    PhiT = Phi_and_phi[:, :, :n]
    phi = Phi_and_phi[:, :, n:]
    L = Minv.matmul(PhiT.transpose(1, 2))
    c = (phi - PhiT.matmul(L)).to_dense()  # TODO: force invertibility
    c_inv = torch.linalg.inv(c)
    B = -L.matmul(c_inv)
    A = (Minv - B.matmul(L.transpose(1, 2))).to_dense()
    Minv_new = to_linear_operator(
        torch.cat((torch.cat((A, B), 2), torch.cat((B.transpose(1, 2), c_inv), 2)), 1)
    )
    coeffs_new = Minv_new.matmul(Y)
    return Minv_new, coeffs_new


def _rbf_regression_mean_and_std(
    train_X: Tensor, train_Y: Tensor, eps: Tensor, coeffs: Tensor, X: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Mean and scale for RBF regression."""
    # NOTE: here, we do not use `KernelLinearOperator` so as to avoid computing
    # distance of `X` to `train_X` twice, one in the linear operator and one in the
    # `idw_weighting` function. Moreover, the linear operator would not be jitted.
    dist = torch.cdist(X, train_X)
    scaled_dist = eps[..., None, None] * dist
    M = torch.scalar_tensor(1.0).addcmul(scaled_dist, scaled_dist).reciprocal()
    mean = M.matmul(coeffs)
    W = torch.maximum(dist.square(), DELTA).reciprocal()
    W_sum_recipr = W.sum(2, keepdim=True).reciprocal()
    V = W.mul(W_sum_recipr)
    std = trace_idw_scale(mean, train_Y, V)
    return mean, std, W_sum_recipr


trace_rbf_regression_mean_and_std = torch.jit.trace(
    _rbf_regression_mean_and_std,
    (
        torch.rand(7, 5, 2),
        torch.rand(7, 5, 1),
        torch.rand(()),
        torch.rand((7, 5, 1)),
        torch.rand((7, 8, 2)),
    ),
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
        """Instantiates an RBF regression model in Global Optimization.

        Parameters
        ----------
        train_X : Tensor
            A `b x m x d`-dim batched tensor of `m` training points with dimension `d`.
        train_Y : Tensor
            A `b x m x 1`-dim batched tensor of `m` training observations.
        eps : float, optional
            Distance-scaling parameter for the RBF kernel, by default `1.0`.
        svd_tol : float, optional
            Tolerance for singular value decomposition for inversion, by default `1e-8`.
        Minv_and_coeffs : tuple of 2 Tensors, optional
            Precomputed inverse of the RBF kernel matrix and coefficients. By default
            `None`, in which case the model is fit anew to the training data. If
            provided, the model is only partially fit to the new data.
        """
        super().__init__()
        eps = torch.scalar_tensor(eps)
        svd_tol = torch.scalar_tensor(svd_tol)
        if Minv_and_coeffs is None:
            Minv, coeffs = _rbf_fit(train_X, train_Y, eps, svd_tol)
        else:
            Minv, coeffs = _rbf_partial_fit(train_X, train_Y, eps, *Minv_and_coeffs)
        self.train_X = train_X
        self.train_Y = train_Y
        self.register_buffer("eps", eps)
        self.register_buffer("svd_tol", svd_tol)
        self.Minv = Minv  # cannot do `self.register_buffer("Minv", Minv)`
        self.register_buffer("coeffs", coeffs)
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, X: Tensor) -> Normal:
        """Computes the RBF regression model.

        Parameters
        ----------
        X : Tensor
            A `b x q x d`-dim batched tensor of `d`-dim design points, where `q` is the
            number of candidate points to estimate, and `b` is the batched regressor
            size.

        Returns
        -------
        Normal
            The normal
        """
        return trace_rbf_regression_mean_and_std(
            self.train_X, self.train_Y, self.eps, self.coeffs, X
        )
