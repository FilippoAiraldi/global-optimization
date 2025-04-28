"""
Implementation of the acquisition function for RBF/IDW-based Global Optimization
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from math import ceil, pi, sqrt
from typing import Callable, Optional

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


def latin_hypercube_with_nonlinear_constraint(
    n: int,
    q: int,
    d: int,
    lb: Tensor,
    ub: Tensor,
    constraint: Callable[[Tensor], Tensor],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    iteration_limit: int = 10_000,
) -> Tensor:
    """Generates `n` samples of `q`-batches in `d` dimensions using scrambled Latin
    Hypercube Sampling, subject to a nonlinear constraint.

    This function iteratively draws a larger batch of Latin Hypercube Samples within
    the axis-aligned box defined by `lb` and `ub`, applies the user-provided
    `constraint`, and retains only those points that satisfy it. Sampling continues
    until at least `n` valid samples are found, then returns the first `n` of them.

    Parameters
    ----------
    n : int
        The number of samples to generate (also the number of strata per dimension)
    q : int
        The number of candidates per q-batch.
    d : int
        The number of dimensions (variables) of the sample space
    lb : Tensor
        The lower bounds of the sample space. Should be a 1D tensor of length `d`.
    ub : Tensor
        The upper bounds of the sample space. Should be a 1D tensor of length `d`.
    constraint : Callable[[Tensor], Tensor]
        A callable that takes a tensor of shape `(n, q, d)` and returns another tensor
        of shape `n`. For each entry, the constraint is deemed satisfied if the output
        is nonnegative.
    device : torch.device, optional
        The desired device for the output tensor. If `None`, it is taken from `lb`.
    dtype : torch.dtype, optional
        The desired data type for the output tensor. If `None`, it is taken from `lb`.
    iteration_limit : int, optional
        The maximum number of iterations to attempt before raising an error. Default is
        `10000`.

    Returns
    -------
    Tensor
        A tensor of shape `(n, q, d)` containing the generated Latin Hypercube Samples.
        Each value is in the range `[lb, ub)`.
    """
    # constant variables
    device = device or lb.device
    dtype = dtype or lb.dtype
    span = ub - lb
    qdim = q * d

    # start the iterative sampling process
    N = n
    n_valid = 0
    iters = 0
    while n_valid < n:
        # generate N LHS samples
        # stratify the [0, 1] interval into N bins
        cut = torch.linspace(0, 1, steps=N + 1, device=device, dtype=dtype)
        lower, upper = cut[:-1], cut[1:]
        # draw one random point per interval for each dimension independently
        u = torch.rand(N, qdim, device=device, dtype=dtype)
        pts_unscaled = lower.view(N, 1) + u * (upper - lower).view(N, 1)
        # generate column-wise permutations and apply them using gather to reorder each
        # column of pts_unscaled by its own permutation
        perm_indices = torch.rand(N, qdim, dtype=dtype, device=device).argsort(0)
        pts_scrambled = pts_unscaled.gather(0, perm_indices)
        # scale to the range [lb, ub)
        pts = lb + span * pts_scrambled.view(N, q, d)

        # compute the number of samples that satisfy the constraint
        constraint_mask = constraint(pts).ge(0)
        n_valid = constraint_mask.sum().item()

        # if n_valid < n, then increase N as follows and repeat the process
        N = ceil(min(20, 1.1 * n / n_valid) * N) if n_valid > 0 else 20 * N

        # if we reach the recursion limit, raise an error
        iters += 1
        if iters > iteration_limit:
            raise RuntimeError(
                f"Recursion limit reached: {iteration_limit}. "
                f"Try increasing the recursion limit or adjusting the constraint."
            )

    return pts[constraint_mask][:n]
