import unittest

import numpy as np

from globopt.core.regression import Array, Idw, RegressorType, fit, partial_fit, predict
from globopt.myopic.acquisition import acquisition as myopic_acquisition
from globopt.nonmyopic.acquisition import acquisition


def naive_acquisition(
    x: Array, mdl: RegressorType, c1: float = 1.5078, c2: float = 1.4246
) -> Array:
    n_samples, horizon, _ = x.shape
    a_ = np.zeros(n_samples)
    for i in range(n_samples):  # <--- this loop is batched in the real implementation
        batch = x[i]
        mdl_ = mdl
        for h in range(horizon):  # <--- this loop cannot be fundamentally batched
            x_ = batch[h].reshape(1, 1, -1)
            y_hat_ = predict(mdl_, x_)
            a_[i] += myopic_acquisition(x_, mdl_, y_hat_, None, c1, c2)
            mdl_ = partial_fit(mdl_, x_, y_hat_)
    return a_


class TestAcquisition(unittest.TestCase):
    def test_acquisition_function__returns_correct_values(self):
        n_var = 3
        n_samples = 10
        h = 5
        X = np.random.randn(1, n_samples, n_var)
        y = np.random.randn(1, n_samples, 1)
        mdl = fit(Idw(), X, y)
        x = np.random.randn(n_samples * 2, h, n_var)

        np.testing.assert_allclose(acquisition(x, mdl), naive_acquisition(x, mdl))


if __name__ == "__main__":
    unittest.main()
