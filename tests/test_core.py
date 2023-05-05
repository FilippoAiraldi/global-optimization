import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # disable jit for testing

import unittest

import numpy as np
from parameterized import parameterized
from scipy.io import loadmat
from sklearn.utils.estimator_checks import check_estimator

from globopt.core.benchmark import get_available_benchmark_tests, get_benchmark_test
from globopt.core.functional_regression import (
    idw_fit,
    idw_partial_fit,
    idw_predict,
    rbf_fit,
    rbf_partial_fit,
    rbf_predict,
)
from globopt.core.regression import IdwRegression, RbfRegression

RESULTS = loadmat(r"tests/data_test_core.mat")


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


class TestRegression(unittest.TestCase):
    def test__with_sklearn_check_estimator(self) -> None:
        check_estimator(IdwRegression())
        check_estimator(RbfRegression())

    @parameterized.expand([(False,), (True,)])
    def test__fit_and_partial_fit(self, use_partial: bool) -> None:
        X = np.array([[-2.61, -1.92, -0.63, 0.38, 2]]).T
        y = f(X).flatten()
        mdls = [
            IdwRegression(),
            RbfRegression("inversequadratic", 0.5),
            RbfRegression("thinplatespline", 0.01),
        ]

        if not use_partial:
            mdls = [mdl.fit(X, y) for mdl in mdls]
        else:
            Xs, ys = np.array_split(X, 3), np.array_split(y, 3)
            mdls = [
                mdl.partial_fit(Xs[0], ys[0])
                .partial_fit(Xs[1], ys[1])
                .partial_fit(Xs[2], ys[2])
                for mdl in mdls
            ]
        x = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_hat = np.asarray([mdl.predict(x) for mdl in mdls])

        np.testing.assert_allclose(y_hat, RESULTS["y_hat"])


class TestFunctionalRegression(unittest.TestCase):
    def test__fit_and_partial_fit(self) -> None:
        X = np.array([[-2.61, -1.92, -0.63, 0.38, 2]]).T
        y = f(X).flatten()
        Xs, ys = np.array_split(X, 3), np.array_split(y, 3)

        fr = [
            idw_fit(Xs[0], ys[0]),
            rbf_fit(Xs[0], ys[0], "inversequadratic", 0.5),
            rbf_fit(Xs[0], ys[0], "thinplatespline", 0.01),
        ]
        for i in range(1, len(Xs)):
            fr[0] = idw_partial_fit(fr[0], Xs[i], ys[i])
            fr[1] = rbf_partial_fit(fr[1], Xs[i], ys[i], "inversequadratic", 0.5)
            fr[2] = rbf_partial_fit(fr[2], Xs[i], ys[i], "thinplatespline", 0.01)

        x = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_hat = np.asarray(
            (
                idw_predict(fr[0], x),
                rbf_predict(fr[1], x, "inversequadratic", 0.5),
                rbf_predict(fr[2], x, "thinplatespline", 0.01),
            )
        )

        np.testing.assert_allclose(y_hat, RESULTS["y_hat"])


class TestBenchmark(unittest.TestCase):
    @parameterized.expand((False, True))
    def test__pareto_set_and_front(self, normalized: bool):
        testnames = get_available_benchmark_tests()
        for name in testnames:
            pbl, _ = get_benchmark_test(name, normalize=normalized)
            ps = pbl.pareto_set()
            pf = pbl.pareto_front()
            pf_actual = pbl.evaluate(ps)
            np.testing.assert_allclose(
                pf_actual.squeeze(), pf.squeeze(), rtol=1e-5, atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
