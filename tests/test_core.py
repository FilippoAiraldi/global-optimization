import unittest

import numpy as np
from parameterized import parameterized
from scipy.io import loadmat

from globopt.core.problems import (
    get_available_benchmark_problems,
    get_benchmark_problem,
)
from globopt.core.regression import Idw, Rbf, fit, partial_fit, predict

RESULTS = loadmat(r"tests/data_test_core.mat")


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
        y_hat = np.concatenate([predict(fr, x) for fr in fitresults], 0)[..., 0]

        np.testing.assert_allclose(y_hat, RESULTS["y_hat"])

    def test__to_str(self) -> None:
        mdl = Rbf("inversemultiquadric", 0.5, svd_tol=0)
        s = mdl.__str__()
        self.assertIsInstance(s, str)
        self.assertIn("kernel=inversemultiquadric", s)
        self.assertIn("eps=0.5", s)
        self.assertIn("svd_tol=0", s)


class TestBenchmark(unittest.TestCase):
    @parameterized.expand((False, True))
    def test__pareto_set_and_front(self, normalized: bool):
        testnames = get_available_benchmark_problems()
        for name in testnames:
            pbl, _ = get_benchmark_problem(name, normalize=normalized)
            ps = pbl.pareto_set()
            pf = pbl.pareto_front()
            pf_actual = pbl.evaluate(ps)
            np.testing.assert_allclose(
                pf_actual.squeeze(), pf.squeeze(), rtol=1e-5, atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
