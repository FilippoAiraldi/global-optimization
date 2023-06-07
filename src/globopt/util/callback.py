"""Callbacks compliant to pymoo API to collect data from algorithm runs."""


from typing import Union

from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback

from globopt.myopic.algorithm import GO, Idw, Rbf, acquisition
from globopt.nonmyopic.algorithm import NonMyopicGO


class BestSoFarCallback(Callback):
    """Callback for storing the best solution found so far."""

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def _update(self, algorithm: Algorithm) -> None:
        self.data["best"].append(algorithm.pop.get("F").min())


class DPStageCostCallback(Callback):
    """
    Callback for storing the dynamic programming's stage cost of choosing a new sampling
    point at each iteration.
    """

    def __init__(self) -> None:
        super().__init__()
        self.data["cost"] = []
        self._prev_regression: Union[None, Idw, Rbf] = None

    def _update(self, algorithm: Algorithm) -> None:
        is_myopic = isinstance(algorithm, GO)
        is_non_myopic = isinstance(algorithm, NonMyopicGO)
        mdl = self._prev_regression
        assert is_myopic or is_non_myopic, "Algorithm is not a GO instance."
        if mdl is not None:
            # save the stage cost, i.e., the acquistion, of having chosen the new sample
            # given the previous regression model
            if is_non_myopic:
                x_new = algorithm.acquisition_min_res.X[None]
                a = acquisition(x_new, mdl, None, None, algorithm.c1, algorithm.c2)
            else:
                a = algorithm.acquisition_min_res.F
            self.data["cost"].append(a.item())
        self._prev_regression = algorithm.regression
