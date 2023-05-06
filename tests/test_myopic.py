import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # disable jit for testing

import pickle
import unittest
from typing import Any

import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from scipy.io import loadmat

from globopt.core.regression import Rbf, fit, predict
from globopt.myopic.acquisition import (
    _idw_distance,
    _idw_variance,
    acquisition,
    idw_weighting,
)
from globopt.myopic.algorithm import GO

RESULTS = loadmat(r"tests/data_test_myopic.mat")
with open(r"tests/data_test_myopic.pkl", "rb") as f:
    RESULTS.update(pickle.load(f))


def f(x):
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


class Simple1DProblem(Problem):
    def __init__(self) -> None:
        super().__init__(n_var=1, n_obj=1, xl=-3, xu=3, type_var=float)

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *_, **__) -> None:
        out["F"] = f(x)

    def _calc_pareto_front(self) -> float:
        return 0.279504

    def _calc_pareto_set(self) -> float:
        return -0.959769


class TestAcquisition(unittest.TestCase):
    def test_acquisition_function__returns_correct_values(self):
        X = np.array([[-2.61, -1.92, -0.63, 0.38, 2]]).T
        y = f(X)

        X = np.expand_dims(X, 0)  # add batch dimension
        y = np.expand_dims(y, 0)
        mdl = fit(Rbf("thinplatespline", 0.01), X, y)
        x = np.linspace(-3, 3, 1000).reshape(1, -1, 1)
        y_hat = predict(mdl, x)
        dym = y.max() - y.min()  # span of observations
        W = idw_weighting(x, X)
        s = _idw_variance(y_hat, y, W)
        z = _idw_distance(W)
        a = acquisition(x, mdl, y_hat, dym, 1, 0.5)

        out = np.concatenate((s, z, a), 0)[..., 0]
        np.testing.assert_allclose(out, RESULTS["acquisitions"])


class TestAlgorithm(unittest.TestCase):
    def test__returns_correct_result(self):
        problem = Simple1DProblem()
        x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
        regression = RbfRegression("thinplatespline", 0.01)
        algorithm = GO(
            regression=regression,
            init_points=x0,
            acquisition_fun_kwargs={"c1": 1, "c2": 0.5},
        )

        res = minimize(
            problem,
            algorithm,
            termination=("n_iter", 6),
            verbose=True,
            seed=1,
            save_history=True,
        )

        x = np.linspace(*problem.bounds(), 500).reshape(-1, 1)
        out: dict[int, tuple[np.ndarray, ...]] = {}
        for i, algo in enumerate(res.history, start=1):
            y_hat = algo.regression.predict(x)
            Xm = algo.pop.get("X").reshape(-1, 1)
            ym = algo.pop.get("F").reshape(-1)
            a = acquisition(x, y_hat, Xm, ym, None, **algo.acquisition_fun_kwargs)
            acq_min = (
                algo.acquisition_min_res.opt.item().X
                if hasattr(algo, "acquisition_min_res")
                else np.nan
            )
            out[i] = (y_hat, Xm, ym, a, acq_min)

        for key in out:
            for actual, expected in zip(out[key], RESULTS[key]):
                np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
