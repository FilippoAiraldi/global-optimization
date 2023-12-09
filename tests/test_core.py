import unittest

import numpy as np
import torch
from parameterized import parameterized
from scipy.io import loadmat

from globopt.core_bt.problems import (
    SimpleProblem,
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
)
from globopt.core_bt.regression import Idw, Rbf

RESULTS = loadmat("tests/data_test_core.mat")


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


class TestRegression(unittest.TestCase):
    def test_fit_and_partial_fit(self) -> None:
        problem = SimpleProblem()
        X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device="cpu").reshape(
            1, -1, 1
        )
        Y = problem(X)
        n = 3
        mdls = [Idw(X[:, :n], Y[:, :n]), Rbf(X[:, :n], Y[:, :n], eps=0.5, svd_tol=0.0)]
        for i in range(len(mdls)):
            mdl = mdls[i]
            if isinstance(mdl, Idw):
                mdls[i] = Idw(X, Y)
            else:
                mdls[i] = Rbf(X, Y, mdl.eps, mdl.svd_tol, (mdl.Minv, mdl.coeffs))
        x_hat = torch.linspace(-3, 3, 100).reshape(1, -1, 1)
        y_hat = torch.cat([mdl(x_hat).mean for mdl in mdls]).squeeze(-1)
        torch.testing.assert_close(
            y_hat, torch.as_tensor(RESULTS["y_hat"][:2], dtype=y_hat.dtype)
        )


EXPECTED_F_OPT: dict[str, float] = {
    # AnotherSimple1dProblem.f.__name__[1:]: -0.669169468,
    SimpleProblem.__name__.lower(): 0.279504,
    # #
    # Ackley.f.__name__[1:]: 0,
    # Adjiman.f.__name__[1:]: -2.02181,
    # Branin.f.__name__[1:]: 0.3978873,
    # CamelSixHumps.f.__name__[1:]: -1.031628453489877,
    # Hartmann3.f.__name__[1:]: -3.86278214782076,
    # Hartmann6.f.__name__[1:]: -3.32236801141551,
    # Himmelblau.f.__name__[1:]: 0,
    # Rosenbrock8.f.__name__[1:]: 0,
    # Step2Function5.f.__name__[1:]: 0,
    # StyblinskiTang5.f.__name__[1:]: -39.16599 * 5,
}


class TestProblems(unittest.TestCase):
    @parameterized.expand(
        [
            (n,)
            for n in get_available_benchmark_problems()
            + get_available_simple_problems()
        ]
    )
    def test_optimal_value_and_point(self, testname: str):
        problem, _, _ = get_benchmark_problem(testname)
        expected = EXPECTED_F_OPT[testname]
        actual = problem._optimal_value
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for x_opt in problem._optimizers:
            f_computed = problem(torch.as_tensor(x_opt))
            expected_ = torch.as_tensor(expected).reshape_as(f_computed)
            torch.testing.assert_close(f_computed, expected_, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
