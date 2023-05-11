import sys
import unittest
from io import StringIO

import numpy as np
from scipy.io import loadmat

from globopt.core.display import PrefixedStream
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


class TestPrefixedStream(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = StringIO()
        sys.stdout = self.capture

    def tearDown(self) -> None:
        sys.stdout = sys.__stdout__

    def test__print_calls__are_prefixed(self):
        prefix = ">>> "
        text1 = "Hello, world!"
        text2 = "Hoe heet je?"
        text3 = "Mijn naam is Filippo."

        print(text1)
        with PrefixedStream.prefixed_print(prefix):
            print(text2)
        print(text3)

        self.assertEqual(
            self.capture.getvalue(), f"{text1}\n{prefix}{text2}\n{text3}\n"
        )


if __name__ == "__main__":
    unittest.main()
