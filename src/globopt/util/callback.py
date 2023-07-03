"""Callbacks compliant to pymoo API to collect data from algorithm runs."""


from typing import Any, Callable, Literal

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
            acq_opt = locals.get("acq_opt")
            if acq_opt is not None:
                self.append(acq_opt)
        else:
            x_new = locals.get("x_new")
            if x_new is not None:
                x_new = x_new.reshape(1, 1, -1)
                mdl = locals["mdl"]
                c1 = locals["c1"]
                c2 = locals["c2"]
                self.append(acquisition(x_new, mdl, c1, c2, None, None).item())


class CallbackCollection:
    """Collection of callbacks."""

    def __init__(
        self, *callbacks: Callable[[Literal["go", "nmgo"], dict[str, Any]], None]
    ) -> None:
        self.callbacks = callbacks

    def __call__(
        self, algorithm: Literal["go", "nmgo"], locals: dict[str, Any]
    ) -> None:
        for callback in self.callbacks:
            callback(algorithm, locals)
