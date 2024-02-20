from typing import Any, Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.multi_step_lookahead import (
    TAcqfArgConstructor,
    qMultiStepLookahead,
)
from botorch.models.model import Model
from botorch.sampling.base import MCSampler

from globopt.sampling import PosteriorMeanSampler


def make_idw_acq_factory(
    c1: float, c2: float, span_Y_min: float = 1e-3
) -> TAcqfArgConstructor:
    """Returns a kwargs factory for `IdwAcquisitionFunction` with the given parameters,
    useful for `qMultiStepLookahead`."""

    def _inner(*_, **__) -> dict[str, Any]:
        return {"c1": c1, "c2": c2, "span_Y_min": span_Y_min}

    return _inner


class qRollout(qMultiStepLookahead):
    """Rollout nonmyopic acquisition function based on `qMultiStepLookahead`.

    This acquisition function rolls out the base acquisition function along the given
    horizon, and returns the sum of the values of the base acquisition function at each
    stage. Rollout is known to always outperform greedy selection, and is a good tool
    for improving the performance of myopic base acquisition functions.
    """

    def __init__(
        self,
        model: Model,
        horizon: int,
        valfunc_cls: type[AcquisitionFunction],  # base policy
        valfunc_argfactory: Optional[TAcqfArgConstructor] = None,
        batch_sizes: Optional[list[int]] = None,  # value of `q` at each stage
        fantasies_sampler: Optional[MCSampler] = None,
        valfunc_sampler: Optional[MCSampler] = None,
    ) -> None:
        """Instantiates the rollout acquisition function.

        Parameters
        ----------
        model : Model
            A fitted model.
        horizon : int
            Length of the rollout horizon. Must be at least 2.
        valfunc_cls : type[AcquisitionFunction]
            The type of the base acquisition function class.
        valfunc_argfactory: TAcqfArgConstructor, optional
            A callable that takes the current model and observatiosn and returns
            the kwargs to pass to the base acquisition function constructor.
        batch_sizes : list[int], optional
            A list `[q_1, ..., q_k]` containing the batch sizes for the `k` look-ahead
            steps. By default, all batch sizes are set to 1 along the horizon.
        fantasies_sampler : MCSampler, optional
            Sampler used to sample the fantasies. By default, `PosterionMeanSampler` is
            used, i.e., a sampler that always takes the posterior mean as the single
            sample.
        valfunc_sampler : MCSampler, optional
            A custom sampler to override the sampling of the base acquisition function
            values.
        """
        # prepare and check inputs
        if horizon < 2:
            raise ValueError("horizon must be at least 2")
        if batch_sizes is None:
            batch_sizes = [1] * (horizon - 1)
        if fantasies_sampler is None:  # by default, sample the posterior mean
            fantasies_sampler = PosteriorMeanSampler()
        no_valfunc_sampler = valfunc_sampler is None
        if no_valfunc_sampler:
            inner_mc_sample = None
        else:
            if len(valfunc_sampler.sample_shape) > 1:
                raise ValueError("`valfunc_sampler` must have a single sample shape")
            inner_mc_sample = valfunc_sampler.sample_shape[0]

        # construct base
        super().__init__(
            model=model,
            batch_sizes=batch_sizes,
            samplers=[fantasies_sampler] * (horizon - 1),
            valfunc_cls=[valfunc_cls] * horizon,
            valfunc_argfacs=[valfunc_argfactory] * horizon,
            inner_mc_samples=[inner_mc_sample] * horizon,
        )

        # override inner samplers post-construction
        if not no_valfunc_sampler:
            new_samplers = []
            for sampler in self.inner_samplers:
                new_samplers.append(None if sampler is None else valfunc_sampler)
            self.inner_samplers = torch.nn.ModuleList(new_samplers)
