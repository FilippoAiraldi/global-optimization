import unittest
from itertools import product

import numpy as np
from parameterized import parameterized
from scipy.io import loadmat

from globopt.core.problems import (
    Ackley,
    Adjiman,
    AnotherSimple1dProblem,
    Branin,
    CamelSixHumps,
    Hartmann3,
    Hartmann6,
    Himmelblau,
    Rosenbrock8,
    Simple1dProblem,
    Step2Function5,
    StyblinskiTang5,
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
)
from globopt.core.regression import Idw, Rbf, fit, partial_fit, predict

RESULTS = loadmat("tests/data_test_core.mat")


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


class TestRegression(unittest.TestCase):
    def test__fit_and_partial_fit(self) -> None:
        X = np.array([-2.61, -1.92, -0.63, 0.38, 2]).reshape(1, -1, 1)
        y = f(X)
        Xs, ys = np.array_split(X, 3, axis=1), np.array_split(y, 3, axis=1)

        mdls = [
            Idw(),
            Rbf("inversequadratic", 0.5, svd_tol=0),
            Rbf("thinplatespline", 0.01, svd_tol=0),
        ]
        fitresults = [fit(mdl, Xs[0], ys[0]) for mdl in mdls]
        for i in range(1, len(Xs)):
            fitresults = [partial_fit(fr, Xs[i], ys[i]) for fr in fitresults]
        x = np.linspace(-3, 3, 100).reshape(1, -1, 1)
        y_hat = np.asarray([predict(fr, x).squeeze() for fr in fitresults])

        np.testing.assert_allclose(y_hat, RESULTS["y_hat"])

    def test__to_str(self) -> None:
        mdl = Rbf("inversemultiquadric", 0.5, svd_tol=0)
        s = mdl.__str__()
        self.assertIsInstance(s, str)
        self.assertIn("kernel=inversemultiquadric", s)
        self.assertIn("eps=0.5", s)
        self.assertIn("svd_tol=0", s)


EXPECTED_F_OPT: dict[str, float] = {
    AnotherSimple1dProblem.f.__name__: -0.669169468,
    Simple1dProblem.f.__name__: 0.279504,
    #
    Ackley.f.__name__: 0,
    Adjiman.f.__name__: -2.02181,
    Branin.f.__name__: 0.3978873,
    CamelSixHumps.f.__name__: -1.031628453489877,
    Hartmann3.f.__name__: -3.86278214782076,
    Hartmann6.f.__name__: -3.32236801141551,
    Himmelblau.f.__name__: 0,
    Rosenbrock8.f.__name__: 0,
    Step2Function5.f.__name__: 0,
    StyblinskiTang5.f.__name__: -39.16599 * 5,
}


class TestProblems(unittest.TestCase):
    @parameterized.expand(
        product(
            get_available_benchmark_problems() + get_available_simple_problems(),
            (True, False),
        )
    )
    def test__f_opt(self, testname: str, normalized: bool):
        problem, _, _ = get_benchmark_problem(testname, normalize=normalized)
        expected = EXPECTED_F_OPT[problem.f.__name__]
        actual = problem.f_opt
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

        x_opt = problem.x_opt
        for i in range(1, x_opt.shape[0]):
            np.testing.assert_allclose(
                problem.f(x_opt[i, np.newaxis]), expected, rtol=1e-5, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
