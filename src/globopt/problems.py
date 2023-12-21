"""
Collection of popular tests for benchmarking optimization algorithms. These tests were
implemented according to [1, 2].

References
----------
[1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
    optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
[2] Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments: Test
    Functions and Datasets. Retrieved May 3, 2023, from
    http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano.
"""

from types import MappingProxyType
from typing import Any, Literal

import torch
from botorch.test_functions import (
    Ackley,
    Branin,
    Hartmann,
    Michalewicz,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang,
)
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


class SimpleProblem(SyntheticTestFunction):
    r"""Simple problem:

        f(x) = (1 + x sin(2x) cos(3x) / (1 + x^2))^2 + x^2 / 12 + x / 10

    x is bounded [-3, +3], and f in has a global minimum at `x_opt = -0.959769`
    with `f_opt = 0.2795`.
    """

    dim = 1
    _optimal_value = 0.279504
    _optimizers = [(-0.959769,)]
    _bounds = [(-3.0, +3.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        X2 = X.square()
        return (
            (1 + X * torch.sin(2 * X) * torch.cos(3 * X) / (1 + X2)).square()
            + X2 / 12
            + X / 10
        )


class Adjiman(SyntheticTestFunction):
    r"""Adjiman function, a 2-dimensional synthetic test function given by:

        f(x) = cos(x) sin(y) - x / (y^2 + 1).

    x is bounded [-1,2], y in [-1,1]. f in has a global minimum at
    `x_opt = (2, 0.10578)` with `f_opt = -2.02181`.
    """

    dim = 2
    _optimal_value = -2.02181
    _optimizers = [(2.0, 0.10578)]
    _bounds = [(-1.0, 2.0), (-1.0, 1.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x = X[..., 0]
        y = X[..., 1]
        return x.cos().mul(y.sin()).addcdiv(x, y.square() + 1.0, value=-1)


TESTS: dict[
    str, tuple[type[SyntheticTestFunction], dict[str, Any], int, Literal["rbf", "idw"]]
] = MappingProxyType(
    {
        problem.__name__.lower(): (problem, kwargs, max_evals, regressor_type)
        for problem, kwargs, max_evals, regressor_type in [
            (Ackley, {}, 50, "idw"),
            (Adjiman, {}, 10, "idw"),
            (Branin, {}, 40, "idw"),
            (Hartmann, {"dim": 3}, 50, "rbf"),
            (Michalewicz, {"dim": 5}, 40, "rbf"),  # untested
            (Rastrigin, {"dim": 4}, 80, "rbf"),
            (Rosenbrock, {"dim": 8}, 50, "rbf"),
            (Shekel, {"m": 7}, 60, "rbf"),
            (SixHumpCamel, {"bounds": [(-5.0, 5.0), (-5.0, 5.0)]}, 10, "rbf"),
            (StyblinskiTang, {"dim": 5}, 60, "rbf"),
        ]
    }
)


def get_available_benchmark_problems() -> list[str]:
    """Gets the names of all the available benchmark test problems.

    Returns
    -------
    list of str
        Names of all the available benchmark tests.
    """
    return list(TESTS.keys())


def get_benchmark_problem(
    name: str,
) -> tuple[SyntheticTestFunction, int, Literal["rbf", "idw"]]:
    """Gets an instance of a benchmark synthetic problem.

    Parameters
    ----------
    name : str
        Name of the benchmark test.

    Returns
    -------
    tuple of (SyntheticTestFunction, int, str)
        The problem, the maximum number of evaluations and the regression type suggested
        for its optimization.

    Raises
    ------
    KeyError
        Raised if the name of the benchmark test is not found.
    """
    cls, kwargs, max_evals, regressor = TESTS[name.lower()]
    return cls(**kwargs), max_evals, regressor
