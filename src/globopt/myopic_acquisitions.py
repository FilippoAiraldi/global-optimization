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

from globopt.regression import Idw, Rbf


def _idw_distance(W_sum_recipr: Tensor) -> Tensor:
    """Computes the IDW distance function.

    Parameters
    ----------
    W_sum_recipr : Tensor
        `(b0 x b1 x ...) x n x 1` reciprocal of the sum of the IDW weights, i.e.,
        `1/sum(W)`. `n` is the number of candidate points whose reciprocal of the sum of
        weights is passed, and `b`s are the batched regressor sizes.

    Returns
    -------
    Tensor
        The standard deviation of shape `(b0 x b1 x ...) x n x 1`.
    """
    return W_sum_recipr.arctan().mul(2 / torch.pi)


def acquisition_function(
    Y_hat: Tensor,
    Y_std: Tensor,
    Y_span: Tensor,
    W_sum_recipr: Tensor,
    c1: Tensor,
    c2: Tensor,
) -> Tensor:
    """Computes the Global Optimization myopic acquisition function.

    Parameters
    ----------
    Y_hat : Tensor
        `(b0 x b1 x ...) x n x 1` estimates of the function values at the candidate
        points. `q` is the number of candidate points, and `b`s are the batched
        regressor sizes.
    Y_std : Tensor
        `(b0 x b1 x ...) x n x 1` standard deviation of the estimates.
    Y_span : Tensor
        `(b0 x b1 x ...) x 1 x 1` span of the training data points, i.e., the difference
        between the maximum and minimum values of these.
    W_sum_recipr : Tensor
        `(b0 x b1 x ...) x n x 1` reciprocal of the sum of the IDW weights, i.e.,
        `1/sum(W)`.
    c1 : scalar Tensor
        Weight of the contribution of the variance function.
    c2 : scalar Tensor
        Weight of the contribution of the distance function.

    Returns
    -------
    Tensor
        Acquisition function of shape `(b0 x b1 x ...) x n x 1`.
    """
    distance = _idw_distance(W_sum_recipr)
    return Y_hat.sub(Y_std, alpha=c1).sub(Y_span.mul(distance), alpha=c2).neg()


class MyopicAcquisitionFunction(AnalyticAcquisitionFunction):
    """Myopic acquisition function for Global Optimization based on RBF/IDW
    regression.

    Computes the myopic acquisition function according to [1] as a function of the
    estimate of the function value at the candidate points, the distance between the
    observed points, and an approximate IDW standard deviation. This acquisition
    does not exploit this deviation to approximate the estimate variance, so it only
    supports `q=1`. For versions that do so, see `qAnalyticalMyopicAcquisitionFunction`
    and `qMcMyopicAcquisitionFunction`.

    Example
    -------
    >>> model = Idw(train_X, train_Y)
    >>> MAF = MyopicAcquisitionFunction(model)
    >>> af = MAF(test_X)

    References
    ----------
    [1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
        functions. Computational Optimization and Applications, 77(2):571–595, 2020
    """

    model: Union[Idw, Rbf]

    def __init__(
        self,
        model: Union[Idw, Rbf],
        c1: Union[float, Tensor],
        c2: Union[float, Tensor],
        **kwargs: Any,
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
        kwargs
            Additional arguments to `AnalyticAcquisitionFunction`.
        """
        super().__init__(model, **kwargs)
        self.register_buffer("c1", torch.scalar_tensor(c1))
        self.register_buffer("c2", torch.scalar_tensor(c2))
        # pre-compute span of training data points
        Y = model.train_Y
        span_Y = Y.max(-2, keepdim=True).values - Y.min(-2, keepdim=True).values
        self.register_buffer("span_Y", span_Y)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        # input `X` is always `b x 1 x d`, and should output tensor of shape `b`
        posterior = self.model.posterior(X.transpose(-3, -2))
        # the posterior's mean, scale and model's W_sum_recipr are all `1 x n x 1`
        return acquisition_function(
            posterior.mean,
            posterior.scale,
            self.span_Y,
            self.model.W_sum_recipr,
            self.c1,
            self.c2,
        ).squeeze((0, 2))
