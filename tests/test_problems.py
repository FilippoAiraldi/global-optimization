import unittest

import torch
from parameterized import parameterized

from globopt.problems import (
    SimpleProblem,
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
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
            expected_ = torch.as_tensor(expected).view_as(f_computed)
            torch.testing.assert_close(f_computed, expected_, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
