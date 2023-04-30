import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # disable jit for testing

import unittest

import numpy as np
from scipy.io import loadmat

from globopt.core.regression import RBFRegression
from globopt.myopic.acquisition import (
    acquisition,
    idw_distance,
    idw_variance,
    idw_weighting,
)

RESULTS = loadmat(r"tests/data_test_myopic.mat")


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


class TestAcquisition(unittest.TestCase):
    def test_acquisition_function__returns_correct_values(self):
        X = np.array([[-2.61, -1.92, -0.63, 0.38, 2]]).T
        y = f(X).flatten()
        mdl = RBFRegression("thinplatespline", 0.01)
        mdl.fit(X, y)

        x = np.linspace(-3, 3, 1000).reshape(-1, 1)
        y_hat = mdl.predict(x)

        dym = y.max() - y.min()  # span of observations
        W = idw_weighting(x, X)
        s = idw_variance(y_hat, y, W)
        z = idw_distance(W)
        a = acquisition(x, y_hat, X, y, dym, 1, 0.5)

        np.testing.assert_allclose(np.stack((s, z, a)), RESULTS["acquisitions"])


if __name__ == "__main__":
    unittest.main()
