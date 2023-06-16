"""Callbacks compliant to pymoo API to collect data from algorithm runs."""


import numpy as np
from vpso.typing import Array1d

from globopt.core.regression import RegressorType
from globopt.myopic.acquisition import acquisition


class BestSoFarCallback(list[float]):
    """Callback for storing the best solution found so far."""

    def __call__(
        self,
        iter: int,
        x_best: Array1d,
        y_best: float,
        x_new: Array1d,
        y_new: float,
        acq_opt: float,
        mdl: RegressorType,
        mdl_new: RegressorType,
    ) -> None:
        self.append(y_best)


class DpStageCostCallback(list[float]):
    """
    Callback for storing the dynamic programming's stage cost of choosing a new sampling
    point at each iteration.
    """

    __slots__ = ("_c1", "_c2", "_is_myopic")

    def __init__(self, c1: float, c2: float, is_myopic: bool = False) -> None:
        self._c1 = c1
        self._c2 = c2
        self._is_myopic = is_myopic

    def __call__(
        self,
        iter: int,
        x_best: Array1d,
        y_best: float,
        x_new: Array1d,
        y_new: float,
        acq_opt: float,
        mdl: RegressorType,
        mdl_new: RegressorType,
    ) -> None:
        if self._is_myopic:
            if not np.isnan(acq_opt):
                self.append(acq_opt)
        elif not np.isnan(x_new):
            self.append(
                acquisition(
                    x_new[np.newaxis, np.newaxis], mdl, None, None, self._c1, self._c2
                ).item()
            )
