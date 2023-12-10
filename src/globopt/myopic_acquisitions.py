"""
Implementation of the acquisition function for RBF/IDW-based Global Optimization
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from typing import Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal

from globopt.regression import Idw, Rbf


def _idw_distance(W_sum_recipr: Tensor) -> Tensor:
    """Computes the IDW distance function.

    Parameters
    ----------
    W_sum_recipr : Tensor
        `b x q x 1` reciprocal of the sum of the IDW weights, i.e., 1/sum(W).

    Returns
    -------
    Tensor
        The standard deviation of shape `b x q x 1`.
    """
    return W_sum_recipr.arctan().mul(2 / torch.pi)


def acquisition_function(
    Y_hat: Tensor,
    Y_std: Tensor,
    Y: Tensor,
    W_sum_recipr: Tensor,
    c1: Tensor,
    c2: Tensor,
) -> Tensor:
    """Computes the IDW myopic acquisition function.

    Parameters
    ----------
    Y_hat : Tensor
        `b x q x 1` estimates of the function values at the candidate points.
    Y_std : Tensor
        `b x q x 1` standard deviation of the estimates.
    Y : Tensor
        `b x m x 1` training data points.
    W_sum_recipr : Tensor
        `b x q x 1` reciprocal of the sum of the IDW weights, i.e., 1/sum(W).
    c1 : Tensor
        Weight of the contribution of the variance function.
    c2 : Tensor
        Weight of the contribution of the distance function.

    Returns
    -------
    Tensor
        Acquisition function of shape `b x q x 1`.
    """
    distance = _idw_distance(W_sum_recipr)
    span = Y.max(1, keepdim=True).values - Y.min(1, keepdim=True).values
    return Y_hat.sub(Y_std, alpha=c1).sub(span.mul(distance), alpha=c2).neg()


trace_acquisition_function = torch.jit.trace(
    acquisition_function,
    (
        torch.rand(7, 6, 1),
        torch.rand(7, 6, 1),
        torch.rand(7, 5, 1),
        torch.rand(7, 6, 1),
        torch.rand(()),
        torch.rand(()),
    ),
)


class GoMyopicAcquisitionFunction(AnalyticAcquisitionFunction):
    """Myopic acquisition function for Global Optimization based on RBF/IDW
    regression. In this form, must be maximized."""

    def __init__(
        self,
        model: Union[Idw, Rbf],
        c1: Union[float, Tensor],
        c2: Union[float, Tensor],
    ) -> None:
        """Instantiates the myopic acquisition function.

        Parameters
        ----------
        model : Idw or Rbf
            BoTorch model based on IDW or RBF regression.
        c1 : float or scalar Tensor
            Weight of the contribution of the variance function.
        c2 : float or scalar Tensor
            Weight of the contribution of the distance function.
        """
        super().__init__(model)
        self.register_buffer("c1", torch.scalar_tensor(c1))
        self.register_buffer("c2", torch.scalar_tensor(c2))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluates the myopic acquisition function on the candidate set X.

        Parameters
        ----------
        X : Tensor
            A `b x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns
        -------
        Tensor
            A `b`-dim tensor of myopic acquisition values at the given
            design points `X`.
        """
        normal: Normal = self.model(X)
        return trace_acquisition_function(
            normal.loc,
            normal.scale,
            self.model.train_Y,
            self.model.W_sum_recipr,
            self.c1,
            self.c2,
        )[:, 0, 0]
