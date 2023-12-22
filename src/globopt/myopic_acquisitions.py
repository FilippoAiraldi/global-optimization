"""
Implementation of the acquisition function for RBF/IDW-based Global Optimization
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from math import pi, sqrt
from typing import Optional, Union

import numpy as np
import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.posteriors import Posterior
from botorch.sampling.base import MCSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor

from globopt.regression import Idw, Rbf, _idw_scale, trace


@trace(torch.rand(2, 3, 4, 1))
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


@trace(
    (
        torch.rand(2, 3, 4, 1),
        torch.rand(2, 3, 4, 1),
        torch.rand(2, 3, 1, 1),
        torch.rand(2, 3, 4, 1),
        torch.rand(()),
        torch.rand(()),
    )
)
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


class GaussHermiteSampler(MCSampler):
    """Sampler for Gauss-Hermite base samples. Supports only a single sample dimension.

    Example
    -------
    >>> sampler = GaussHermiteSampler(1000)
    >>> posterior = model.posterior(test_X)
    >>> samples = sampler(posterior)
    """

    def __init__(self, sample_shape: torch.Size) -> None:
        assert len(sample_shape) == 1, "Only a single dimension is supported."
        super().__init__(sample_shape)
        self.register_buffer("base_weights", None)

    def forward(self, posterior: Posterior) -> Tensor:
        self._construct_base_samples(posterior)
        base_samples = self.base_samples.expand(
            self._get_extended_base_sample_shape(posterior)
        )
        return posterior.rsample_from_base_samples(self.sample_shape, base_samples)

    def _construct_base_samples(self, posterior: Posterior) -> None:
        target_shape = self._get_collapsed_shape(posterior)
        if (
            self.base_samples is not None
            and self.base_weights is not None
            and self.base_samples.shape == target_shape
        ):
            return
        out_dim = target_shape[len(self.sample_shape) :].numel()
        assert out_dim == 1, f"Only output_dim = 1 is supported, but got {out_dim}."
        abscissas, weights = np.polynomial.hermite.hermgauss(self.sample_shape.numel())
        abscissas *= sqrt(2.0)
        weights /= sqrt(pi)
        base_samples = torch.from_numpy(abscissas).view(target_shape)
        base_weights = torch.from_numpy(weights).view(target_shape)
        self.register_buffer("base_samples", base_samples)
        self.register_buffer("base_weights", base_weights)
        self.to(device=posterior.device, dtype=posterior.dtype)


class IdwAcquisitionFunction(AnalyticAcquisitionFunction):
    """IDW (myopic) acquisition function for Global Optimization based on RBF/IDW
    regression.

    Computes the myopic acquisition function according to [1] as a function of the
    estimate of the function value at the candidate points, the distance between the
    observed points, and an approximate IDW standard deviation. This acquisition
    does not exploit this deviation to approximate the estimate variance, and it only
    supports `q = 1`. For a version that does so instead, see `qIdwAcquisitionFunction`.

    Example
    -------
    >>> model = Idw(train_X, train_Y)
    >>> MAF = IdwAcquisitionFunction(model)
    >>> af = MAF(test_X)

    References
    ----------
    [1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
        functions. Computational Optimization and Applications, 77(2):571–595, 2020
    """

    def __init__(
        self,
        model: Union[Idw, Rbf],
        c1: Union[float, Tensor],
        c2: Union[float, Tensor],
        span_Y_min: float = 1e-3,
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
        span_Y_min: float, optional
            Minimum value of the span of observed values (avoids that the distance
            contribution is null).
        """
        super().__init__(model)
        Y_min, Y_max = model.train_Y.aminmax(dim=-2, keepdim=True)
        self.register_buffer("span_Y", (Y_max - Y_min).clamp_min(span_Y_min))
        self.register_buffer("c1", torch.scalar_tensor(c1))
        self.register_buffer("c2", torch.scalar_tensor(c2))

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


class qIdwAcquisitionFunction(MCAcquisitionFunction):
    """Sampling-based myopic acquisition function for Global Optimization based on
    RBF/IDW regression.

    In contrast to `IdwAcquisitionFunction`, this acquisition function approximates the
    expected value of the acquisition function via sampling techniques, such as (Quasi)
    Monte Carlo or Gauss-Hermite quadrature. This allows to exploit the IDW standard
    deviation to better take into account the uncertainty in the regression estimate. It
    supports `q > 1`.

    Example
    -------
    >>> model = Idw(train_X, train_Y)
    >>> sampler = SobolQMCNormalSampler(1024)
    >>> McMAF = qIdwAcquisitionFunction(model, sampler)
    >>> af = McMAF(test_X)
    """

    def __init__(
        self,
        model: Union[Idw, Rbf],
        c1: Union[float, Tensor],
        c2: Union[float, Tensor],
        sampler: Optional[MCSampler],
        span_Y_min: float = 1e-3,
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
        span_Y_min: float, optional
            Minimum value of the span of observed values (avoids that the distance
            contribution is null).
        """
        super().__init__(model, sampler)
        Y_min, Y_max = model.train_Y.aminmax(dim=-2, keepdim=True)
        self.register_buffer("span_Y", (Y_max - Y_min).clamp_min(span_Y_min))
        self.register_buffer("c1", torch.scalar_tensor(c1))
        self.register_buffer("c2", torch.scalar_tensor(c2))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # input of this forward is `b x q x d`, and output `b`. See the note in
        # regression.py to understand these shapes. Mostly, we use `q = 1`, in that case
        # the posterior can be interpreted as `b` independent normal distributions.

        # NOTE: there is no need to use sampling to estimate all the terms of the
        # acquisition function, but only for the scale. This is because this term is the
        # only one that depends on the posterior sample variances (actually, it is the
        # formula of the  variance itself), and we cannot directly compute. The rest
        # are either deterministic or have an analytical expression.
        mdl = self.model
        sampler = self.sampler
        posterior = mdl.posterior(X)

        samples = self.get_posterior_samples(posterior)
        scale = _idw_scale(samples, mdl.train_Y, posterior._V)
        if hasattr(sampler, "base_weights") and sampler.base_weights is not None:
            scale = sampler.base_weights.unsqueeze(-1).mul(scale).sum(0, keepdim=True)

        acqvals = acquisition_function(
            posterior.mean,  # `b x q x 1`
            scale,  # `n_samples x b x q x 1` or `1 x b x q x 1` for GH quadrature
            self.span_Y,  # `b x 1 x 1` or # `1 x 1`
            posterior._W_sum_recipr,  # `b x q x 1`
            self.c1,
            self.c2,
        )
        return acqvals.amax((-2, -1)).mean(0)
