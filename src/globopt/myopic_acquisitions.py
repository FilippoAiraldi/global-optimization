"""
Implementation of the acquisition function for RBF/IDW-based Global Optimization
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from typing import Optional, Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.sampling.base import MCSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor

from globopt.regression import Idw, Rbf, _idw_scale


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


class AcquisitionFunctionMixin:
    """Mixin class for acquisition functions based on RBF/IDW regression."""

    def _setup(
        self,
        c1: Union[float, Tensor],
        c2: Union[float, Tensor],
        accept_batch_regression: bool,
        span_Y_min: float = 1e-3,
    ) -> None:
        """Setups the acquisition function buffers, and performs some checks."""
        if not accept_batch_regression:
            for tensor in (self.model.train_X, self.model.train_Y):
                if tensor.ndim >= 3 and any(s != 1 for s in tensor.shape[:-2]):
                    raise ValueError(
                        "Expected non-batched regression; got training data with shape "
                        + str(tensor.shape)
                    )
        Y_min, Y_max = self.model.train_Y.aminmax(dim=-2, keepdim=True)
        self.register_buffer("span_Y", (Y_max - Y_min).clamp_min(span_Y_min))
        self.register_buffer("c1", torch.scalar_tensor(c1))
        self.register_buffer("c2", torch.scalar_tensor(c2))


class MyopicAcquisitionFunction(AnalyticAcquisitionFunction, AcquisitionFunctionMixin):
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
        self, model: Union[Idw, Rbf], c1: Union[float, Tensor], c2: Union[float, Tensor]
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
        self._setup(c1, c2, accept_batch_regression=False)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        # input of this forward is `b x 1 x d`, and output `b`
        posterior = self.model.posterior(X.transpose(-3, -2))
        return acquisition_function(
            posterior.mean,  # `1 x n x 1`
            posterior._scale,  # `1 x n x 1`
            self.span_Y,  # `1 x 1 x 1` or # `1 x 1`
            posterior._W_sum_recipr,  # `1 x n x 1`
            self.c1,
            self.c2,
        ).squeeze((0, 2))


class qMcMyopicAcquisitionFunction(MCAcquisitionFunction, AcquisitionFunctionMixin):
    """Monte Carlo-based myopic acquisition function for Global Optimization based on
    RBF/IDW regression.

    In contrast to `qAnalyticalMyopicAcquisitionFunction`, this acquisition function
    does not approximate the expectation w.r.t. the IDW variance, and instead uses
    Monte Carlo sampling to compute the expectation of the acquisition values

    Example
    -------
    >>> model = Idw(train_X, train_Y)
    >>> sampler = SobolQMCNormalSampler(1024)
    >>> McMAF = qMcMyopicAcquisitionFunction(model, sampler)
    >>> af = McMAF(test_X)
    """

    def __init__(
        self,
        model: Union[Idw, Rbf],
        c1: Union[float, Tensor],
        c2: Union[float, Tensor],
        sampler: Optional[MCSampler],
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
        sampler : MCSampler, optional
            The sampler used to draw base samples.
        """
        super().__init__(model, sampler)
        self._setup(c1, c2, accept_batch_regression=False)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # input of this forward is `b x q x d`, and output `b`. See the note in
        # regression.py to understand these shapes.
        # first, compute the posterior for X, and sample from it
        mdl = self.model
        posterior = mdl.posterior(X)
        samples = self.get_posterior_samples(posterior)

        # then, for each sample, compute its mean prediction and its IDW scale, and
        # finally, compute acquisition and reduce in t- and q-batches
        scale = _idw_scale(samples, mdl.train_Y, posterior._V)
        acqvals = acquisition_function(
            samples, scale, self.span_Y, posterior._W_sum_recipr, self.c1, self.c2
        )
        return acqvals.amax((-2, -1)).mean(0)  # tuple(range(acqvals.ndim - 2))
