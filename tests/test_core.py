import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # disable jit for testing

import unittest

import numpy as np
from parameterized import parameterized
from scipy.io import loadmat
from sklearn.utils.estimator_checks import check_estimator

from globopt.core.regression import IDWRegression, RBFRegression

RESULTS = loadmat(r"tests/data_test_core.mat")


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


class TestIDWandRBFRegression(unittest.TestCase):
    def test__with_sklearn_check_estimator(self) -> None:
        check_estimator(IDWRegression())
        check_estimator(RBFRegression())

    @parameterized.expand([(False,), (True,)])
    def test__fit_and_partial_fit(self, use_partial: bool) -> None:
        X = np.array([[-2.61, -1.92, -0.63, 0.38, 2]]).T
        y = f(X).flatten()
        mdls = [
            IDWRegression(),
            RBFRegression("inversequadratic", 0.5),
            RBFRegression("thinplatespline", 0.01),
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


if __name__ == "__main__":
    unittest.main()
