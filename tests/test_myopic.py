import pickle
import unittest

import numpy as np
from vpso.typing import Array1d

from globopt.core.problems import simple1dproblem
from globopt.core.regression import Rbf, fit, predict
from globopt.myopic.acquisition import (
    _idw_distance,
    _idw_variance,
    _idw_weighting,
    acquisition,
)
from globopt.myopic.algorithm2 import go

with open(r"tests/data_test_myopic.pkl", "rb") as f:
    RESULTS = pickle.load(f)


def f(x: np.ndarray) -> np.ndarray:
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


class TestAcquisition(unittest.TestCase):
    def test__returns_correct_values(self):
        X = np.array([-2.61, -1.92, -0.63, 0.38, 2]).reshape(1, -1, 1)
        y = f(X)

        mdl = fit(Rbf("thinplatespline", 0.01, svd_tol=0), X, y)
        x = np.linspace(-3, 3, 1000).reshape(1, -1, 1)
        y_hat = predict(mdl, x)
        dym = y.ptp((1, 2), keepdims=True)
        W = _idw_weighting(x, X, mdl.exp_weighting)
        s = _idw_variance(y_hat, y, W)
        z = _idw_distance(W)
        a = acquisition(x, mdl, y_hat, dym, 1, 0.5)

        out = np.asarray((s.squeeze(), z.squeeze(), a.squeeze()))
        np.testing.assert_allclose(out, RESULTS["acquisitions"], atol=1e-6, rtol=1e-6)


class TestAlgorithm(unittest.TestCase):
    def test__returns_correct_result(self):
        f = simple1dproblem.f
        lb = simple1dproblem.lb
        ub = simple1dproblem.ub
        x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
        c1 = 1
        c2 = 0.5
        history: list[tuple[Array1d, ...]] = []
        x = np.linspace(lb, ub, 500)

        def save_history(
            iter: int,
            x_best: Array1d,
            y_best: float,
            x_new: Array1d,
            y_new: float,
            a_opt: float,
            mdl: Rbf,
            mdl_new: Rbf,
        ) -> None:
            if iter > 0:
                x_ = x.reshape(1, -1, 1)
                y_hat = predict(mdl, x_)
                a = acquisition(x_, mdl, y_hat, None, c1, c2)
                history.append((y_hat, mdl.Xm_, mdl.ym_, a, x_new))

        # run the optimization
        go(
            func=f,
            lb=lb,
            ub=ub,
            mdl=Rbf("thinplatespline", 0.01),
            init_points=x0,
            c1=c1,
            c2=c2,
            maxiter=6,
            seed=1909,
            callback=save_history,
        )

        out = dict(enumerate(history, start=1))
        for key in out:
            for actual, expected in zip(out[key], RESULTS[key]):
                np.testing.assert_allclose(
                    np.squeeze(actual), np.squeeze(expected), atol=1e-4, rtol=1e-4
                )


if __name__ == "__main__":
    unittest.main()
