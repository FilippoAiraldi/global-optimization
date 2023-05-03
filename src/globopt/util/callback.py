"""Callbacks compliant to pymoo API to collect data from algorithm runs."""


from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback


class BestSoFarCallback(Callback):
    """Callback for storing the best solution found so far."""

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm: Algorithm) -> None:
        self.data["best"].append(algorithm.pop.get("F").min())
