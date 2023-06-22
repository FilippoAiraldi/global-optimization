import pickle
import unittest
from typing import Any, Literal

import numpy as np
from vpso.typing import Array1d

from globopt.core.problems import Simple1dProblem
from globopt.core.regression import Kernel, Rbf, fit, predict
from globopt.myopic.acquisition import (
    _idw_distance,
    _idw_variance,
    _idw_weighting,
    acquisition,
)
from globopt.myopic.algorithm import go

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

        mdl = fit(Rbf(Kernel.ThinPlateSpline, 0.01, svd_tol=0.0), X, y)
        x = np.linspace(-3, 3, 1000).reshape(1, -1, 1)
        y_hat = predict(mdl, x)
        dym = y.ptp(1, keepdims=True)
        W = _idw_weighting(x, X, mdl.exp_weighting)
        s = _idw_variance(y_hat, y, W)
        z = _idw_distance(W)
        a = acquisition(x, mdl, 1.0, 0.5, y_hat, dym)

        out = np.asarray((s.squeeze(), z.squeeze(), a.squeeze()))
        np.testing.assert_allclose(out, RESULTS["acquisitions"], atol=1e-6, rtol=1e-6)


class TestAlgorithm(unittest.TestCase):
    def test__returns_correct_result(self):
        f = Simple1dProblem.f
        lb = Simple1dProblem.lb
        ub = Simple1dProblem.ub
        x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
        c1 = 1.0
        c2 = 0.5
        history: list[tuple[Array1d, ...]] = []
        x = np.linspace(lb, ub, 500)

        def save_history(_: Literal["go", "nmgo"], locals: dict[str, Any]) -> None:
            if locals.get("iteration", 0) > 0:
                x_ = x.reshape(1, -1, 1)
                mdl = locals["mdl"]
                y_hat = predict(mdl, x_)
                a = acquisition(x_, mdl, c1, c2, y_hat, None)
                history.append((y_hat, mdl.Xm_, mdl.ym_, a, locals["x_new"]))

        # run the optimization
        go(
            func=f,
            lb=lb,
            ub=ub,
            mdl=Rbf(Kernel.ThinPlateSpline, 0.01),
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
