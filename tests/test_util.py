import unittest
from typing import Literal
from unittest.mock import Mock

import numpy as np
from parameterized import parameterized
from pymoo.util.normalization import ZeroToOneNormalization

from globopt.core.problems import Adjiman, Simple1dProblem
from globopt.core.regression import Kernel, Rbf
from globopt.myopic.algorithm import go
from globopt.nonmyopic.algorithm import nmgo
from globopt.util.callback import (
    BestSoFarCallback,
    CallbackCollection,
    DpStageCostCallback,
)
from globopt.util.normalization import backward, forward, normalize_problem


class TestNormalization(unittest.TestCase):
    def test__normalizes_correctly(self) -> None:
        dim = 3
        ub = np.abs(np.random.randn(dim)) + 1
        lb = -np.abs(np.random.randn(dim)) - 1
        ub_new = np.random.randn(dim) ** 2 + 1
        lb_new = -(np.random.randn(dim) ** 2) - 1
        x = np.random.randn(100, dim)

        normalization = ZeroToOneNormalization(lb, ub)
        x_normalized = (
            forward(x, ub, lb, ub_new, lb_new),
            normalization.forward(x) * (ub_new - lb_new) + lb_new,
        )
        x_denormalized = (
            backward(x, ub, lb, ub_new, lb_new),
            normalization.backward((x - lb_new) / (ub_new - lb_new)),
        )

        np.testing.assert_allclose(x_normalized[0], x_normalized[1])
        np.testing.assert_allclose(x_denormalized[0], x_denormalized[1])

    def test__normalizes_x_opt_correctly(self):
        normalized_adjiman = normalize_problem(Adjiman)
        f_opt = Adjiman.f_opt
        self.assertEqual(normalized_adjiman.dim, Adjiman.dim)
        self.assertEqual(normalized_adjiman.lb.shape, Adjiman.lb.shape)
        self.assertEqual(normalized_adjiman.ub.shape, Adjiman.ub.shape)
        self.assertEqual(normalized_adjiman.f_opt, Adjiman.f_opt)
        np.testing.assert_allclose(Adjiman.x_opt, [[2, 0.10578]])
        np.testing.assert_allclose(normalized_adjiman.x_opt, [[1, 0.10578]])
        np.testing.assert_allclose(Adjiman.f(Adjiman.x_opt), f_opt, atol=1e-4)
        np.testing.assert_allclose(
            normalized_adjiman.f(normalized_adjiman.x_opt), f_opt, atol=1e-4
        )


class TestCallback(unittest.TestCase):
    def test__best_so_far_callback(self):
        callback = BestSoFarCallback()
        go(
            func=Simple1dProblem.f,
            lb=Simple1dProblem.lb,
            ub=Simple1dProblem.ub,
            mdl=Rbf(Kernel.ThinPlateSpline, 0.01),
            init_points=[-2.62, -1.2, 0.14, 1.1, 2.82],
            c1=1,
            c2=0.5,
            maxiter=6,
            seed=1909,
            callback=callback,
        )
        np.testing.assert_allclose(
            callback,
            [0.4929, 0.4929, 0.4929, 0.3889, 0.2810, 0.2810, 0.2810],
            atol=1e-4,
            rtol=1e-4,
        )

    @parameterized.expand([("go",), ("nmgo",)])
    def test__dp_stage_cost_callback(self, algorithm: Literal["go", "nmgo"]):
        callback = DpStageCostCallback()
        kwargs = {
            "func": Simple1dProblem.f,
            "lb": Simple1dProblem.lb,
            "ub": Simple1dProblem.ub,
            "mdl": Rbf(Kernel.ThinPlateSpline, 0.01, svd_tol=0.0),
            "init_points": [-2.62, -1.2, 0.14, 1.1, 2.82],
            "c1": 1,
            "c2": 0.5,
            "maxiter": 5,
            "seed": 1909,
        }
        if algorithm == "go":
            expected = [
                0.1715646882008188,
                0.2011653428286008,
                0.24427501622036624,
                0.21051260043554526,
                0.19794195758066008,
            ]
            func = go
        else:
            expected = [
                0.17156476954630534,
                0.20123173223310498,
                0.24415728404033094,
                0.251462332868397,
                0.0719124302371252,
            ]
            func = nmgo
            kwargs.update(
                {"horizon": 2, "discount": 0.9, "rollout": True, "mc_iters": 0}
            )

        func(**kwargs, callback=callback)

        np.testing.assert_allclose(callback, expected, atol=1e-1, rtol=1e-1)

    def test_callback_collection(self):
        callbacks = [Mock() for _ in range(10)]
        collection = CallbackCollection(*callbacks)
        algorithm = object()
        locals = object()

        collection(algorithm, locals)

        for callback in callbacks:
            callback.assert_called_once_with(algorithm, locals)


if __name__ == "__main__":
    unittest.main()
