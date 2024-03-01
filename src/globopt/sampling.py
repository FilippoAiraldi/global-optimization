"""
Implementation of the acquisition function for RBF/IDW-based Global Optimization
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from math import pi, sqrt

import numpy as np
import torch
from botorch.posteriors import Posterior
from botorch.sampling.base import MCSampler
from torch import Tensor


class GaussHermiteSampler(MCSampler):
    """Sampler for Gauss-Hermite base samples. Supports only a single sample dimension.

    Example
    -------
    >>> sampler = GaussHermiteSampler(torch.Size([1000]))
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
