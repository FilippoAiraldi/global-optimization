import unittest

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.normalization import ZeroToOneNormalization

from globopt.core.problems import adjiman, Simple1DProblem
from globopt.core.regression import Rbf
from globopt.myopic.algorithm import GO
from globopt.util.callback import BestSoFarCallback, DPStageCostCallback
from globopt.util.normalization import forward, backward, normalize_problem


class TestNormalization(unittest.TestCase):
    def test_range_normalization__normalizes_correctly(self) -> None:
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

    def test_normalized_problem_wrapper__pareto_set__and__evaluate(self):
        normalized_adjiman = normalize_problem(adjiman)
        f_opt = adjiman.f_opt
        self.assertEqual(normalized_adjiman.dim, adjiman.dim)
        self.assertEqual(normalized_adjiman.lb.shape, adjiman.lb.shape)
        self.assertEqual(normalized_adjiman.ub.shape, adjiman.ub.shape)
        self.assertEqual(normalized_adjiman.f_opt, adjiman.f_opt)
        np.testing.assert_allclose(adjiman.x_opt, [[2, 0.10578]])
        np.testing.assert_allclose(normalized_adjiman.x_opt, [[1, 0.10578]])
        np.testing.assert_allclose(adjiman.f(adjiman.x_opt), f_opt, atol=1e-4)
        np.testing.assert_allclose(
            normalized_adjiman.f(normalized_adjiman.x_opt), f_opt, atol=1e-4
        )


class TestWrapper(unittest.TestCase):
    def test_wrapper(self):
        class Object:
            pass

        fieldvalue = object()
        obj = Object()
        obj.field = fieldvalue
        wrapped_obj = Wrapper(obj)
        self.assertIsNot(obj, wrapped_obj)
        self.assertIs(obj, wrapped_obj.unwrapped)
        self.assertIs(obj, wrapped_obj.wrapped)
        self.assertEqual(obj.field, wrapped_obj.field)


class TestCallback(unittest.TestCase):
    def test__best_so_far_callback(self):
        problem = get_problem("sphere")
        algorithm = GA(pop_size=50)
        res = minimize(
            problem,
            algorithm,
            ("n_gen", 10),
            callback=BestSoFarCallback(),
            save_history=True,
        )
        np.testing.assert_array_equal(
            res.algorithm.callback.data["best"],
            [e.opt.get("F").item() for e in res.history],
        )

    def test__dp_stage_cost_callback(self):
        problem = Simple1DProblem()
        x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
        algorithm = GO(
            regression=Rbf("thinplatespline", 0.01, svd_tol=0),
            init_points=x0,
            c1=1,
            c2=0.5,
            acquisition_min_kwargs={"verbose": True},
        )
        callback = DPStageCostCallback()
        minimize(
            problem,
            algorithm,
            termination=("n_iter", 6),
            verbose=False,
            copy_algorithm=False,
            seed=1,
            callback=callback,
        )
        expected = [
            0.1715646882008188,
            0.2011653428286008,
            0.24427501622036624,
            0.21051260043554526,
            0.19794195758066008,
        ]
        np.testing.assert_allclose(callback.data["cost"], expected)


if __name__ == "__main__":
    unittest.main()
