from collections.abc import Collection
from typing import Any, Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.multi_step_lookahead import (
    TAcqfArgConstructor,
    qMultiStepLookahead,
)
from botorch.models.model import Model
from botorch.sampling.base import MCSampler


def make_idw_acq_factory(
    c1: float, c2: float, span_Y_min: float = 1e-3
) -> TAcqfArgConstructor:
    """Returns a kwargs factory for `IdwAcquisitionFunction` with the given parameters,
    useful for `qMultiStepLookahead`."""

    def _inner(*_, **__) -> dict[str, Any]:
        return {"c1": c1, "c2": c2, "span_Y_min": span_Y_min}

    return _inner


class Ms(qMultiStepLookahead):
    """Multi-step nonmyopic acquisition function based on `qMultiStepLookahead`.

    This acquisition function rolls out the base acquisition function along the given
    horizon and with the provided number of fantasies, and returns the sum of the values
    of the base acquisition function at each stage and fantasy. Rollout is known to
    always outperform greedy selection, and is a good tool for improving the performance
    of myopic base acquisition functions."""

    def __init__(
        self,
        model: Model,
        fantasies_samplers: Collection[MCSampler],
        valfunc_cls: type[AcquisitionFunction],  # base policy
        valfunc_argfactory: Optional[TAcqfArgConstructor] = None,
        valfunc_sampler: Optional[MCSampler] = None,
    ) -> None:
        """Instantiates the multi-step acquisition function.

        Parameters
        ----------
        model : Model
            A fitted model.
        fantasies_samplers : collection of MCSampler
            A collection of samplers, one for each stage of the lookahead. Each sampler
            is used to sample the fantasies for the corresponding stage. The horizon is
            determined by the length of this collection plus one.
        valfunc_cls : type[AcquisitionFunction]
            The type of the base acquisition function class.
        valfunc_argfactory: TAcqfArgConstructor, optional
            A callable that takes the current model and observatiosn and returns
            the kwargs to pass to the base acquisition function constructor.
        valfunc_sampler : MCSampler, optional
            A custom sampler to override the sampling of the base acquisition function
            values (different from sampling the fantasies).
        """
        horizon = len(fantasies_samplers) + 1
        if horizon < 2:
            raise ValueError("horizon must be at least 2")

        # this arg allows to override the default Sobol MC samplers that are used to
        # sample the base acquisition function values, NOT the fantasies
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
            batch_sizes=[1] * (horizon - 1),
            samplers=fantasies_samplers,
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
