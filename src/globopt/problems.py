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

from typing import Literal, Optional

import torch
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
    _check_grad_at_opt: bool = True

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        super().__init__(noise_std, negate, [(-3.0, +3.0)])

    def evaluate_true(self, X: Tensor) -> Tensor:
        X2 = X.square()
        return (
            (1 + X * torch.sin(2 * X) * torch.cos(3 * X) / (1 + X2)).square()
            + X2 / 12
            + X / 10
        )


TESTS: dict[str, tuple[SyntheticTestFunction, int, Literal["rbf", "idw"]]] = {
    problem.__class__.__name__.lower(): (problem, max_evals, regressor_type)
    for problem, max_evals, regressor_type in [(SimpleProblem(), 20, "rbf")]
}


def get_available_benchmark_problems() -> list[str]:
    """Gets the names of all the available benchmark test problems.

    Returns
    -------
    list of str
        Names of all the available benchmark tests.
    """
    return sorted(TESTS.keys() - get_available_simple_problems())


def get_available_simple_problems() -> list[str]:
    """Gets the names of all the simple test problems.

    Returns
    -------
    list of str
        Names of all the available simpler tests.
    """
    return [SimpleProblem.__name__.lower()]


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
    return TESTS[name.lower()]
