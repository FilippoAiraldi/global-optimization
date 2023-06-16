"""Callbacks compliant to pymoo API to collect data from algorithm runs."""


from typing import Any, Literal

import numpy as np

from globopt.myopic.acquisition import acquisition


class BestSoFarCallback(list[float]):
    """Callback for storing the best solution found so far."""

    def __call__(self, _: Literal["go", "nmgo"], locals: dict[str, Any]) -> None:
        self.append(locals["y_best"])


class DpStageCostCallback(list[float]):
    """
    Callback for storing the dynamic programming's stage cost of choosing a new sampling
    point at each iteration.
    """

    def __call__(
        self, algorithm: Literal["go", "nmgo"], locals: dict[str, Any]
    ) -> None:
        if algorithm == "go":
            acq_opt = locals.get("acq_opt", np.nan)
            if not np.isnan(acq_opt):
                self.append(acq_opt)
        else:
            x_new = locals.get("x_new", np.nan)
            if not np.isnan(x_new):
                self.append(
                    acquisition(
                        x_new.reshape(1, 1, -1),
                        locals["mdl"],
                        None,
                        None,
                        locals["c1"],
                        locals["c2"],
                    ).item()
                )
