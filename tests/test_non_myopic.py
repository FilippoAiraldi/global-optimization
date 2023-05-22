import pickle
import unittest

import numpy as np
from parameterized import parameterized
from pymoo.optimize import minimize

from globopt.core.problems import Simple1DProblem
from globopt.core.regression import (
    Array,
    Idw,
    Rbf,
    RegressorType,
    fit,
    partial_fit,
    predict,
)
from globopt.myopic.acquisition import acquisition as myopic_acquisition
from globopt.nonmyopic.acquisition import acquisition, optimal_acquisition
from globopt.nonmyopic.algorithm import NonMyopicGO

with open(r"tests/data_test_non_myopic.pkl", "rb") as f:
    RESULTS = pickle.load(f)

# with open(r"tests/data_test_non_myopic.pkl", "wb") as f:
#     pickle.dump({**RESULTS, "optimal_acquisition_2": nonmyopic_a}, f)


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


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


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

    @parameterized.expand([(1,), (2,)])
    def test_optimal_acquisition__returns_correct_values(self, h: int):
        xl, xu = -3, +3
        c1, c2 = 1.0, 0.5
        f = Simple1DProblem.f
        X = np.array([-2.62, -1.99, 0.14, 1.01, 2.62]).reshape(1, -1, 1)
        y = f(X)
        mdl = fit(Rbf("thinplatespline", 0.01), X, y)
        x = np.linspace(xl, xu, 100).reshape(1, -1, 1)  # add batch dim

        nonmyopic_a = optimal_acquisition(x[0], mdl, h, c1, c2, brute_force=True)

        if h == 1:
            myopic_a = myopic_acquisition(x, mdl, None, None, c1, c2).flatten()
            np.testing.assert_allclose(nonmyopic_a, myopic_a)
        else:
            np.testing.assert_allclose(
                nonmyopic_a, RESULTS[f"optimal_acquisition_{h}"], atol=1e-3, rtol=1e-3
            )


class TestAlgorithm(unittest.TestCase):
    def test__returns_correct_result(self):
        problem = Simple1DProblem()
        x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
        algorithm = NonMyopicGO(
            horizon=2,
            regression=Rbf("thinplatespline", 0.01, svd_tol=0),
            init_points=x0,
            acquisition_fun_kwargs={"c1": 1, "c2": 0.5},
        )

        res = minimize(
            problem,
            algorithm,
            termination=("n_iter", 4),
            seed=1,
            save_history=True,
        )

        x = np.linspace(*problem.bounds(), 500).reshape(-1, 1)
        out: dict[int, tuple[np.ndarray, ...]] = {}
        for i, algo in enumerate(res.history, start=1):
            y_hat = predict(algo.regression, x[np.newaxis])
            Xm = algo.pop.get("X").reshape(-1, 1)
            ym = algo.pop.get("F").reshape(-1)
            a = acquisition(
                x.reshape(-1, 1, 1), algo.regression, **algo.acquisition_fun_kwargs
            )
            acq_min = (
                algo.acquisition_min_res.opt.item().X
                if hasattr(algo, "acquisition_min_res")
                else np.nan
            )
            out[i] = (y_hat.squeeze(), Xm, ym, a, acq_min)

        for key in out:
            for actual, expected in zip(out[key], RESULTS[key]):
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
