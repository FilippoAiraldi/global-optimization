import unittest
from copy import deepcopy

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.normalization import ZeroToOneNormalization

from globopt.core.problems import Adjiman
from globopt.util.callback import BestSoFarCallback
from globopt.util.normalization import NormalizedProblemWrapper, RangeNormalization
from globopt.util.wrapper import Wrapper


class TestNormalization(unittest.TestCase):
    def test_range_normalization__normalizes_correctly(self) -> None:
        xu = np.random.randn() ** 2
        xl = -xu
        xu_new = np.random.randn() ** 2
        xl_new = -xu_new

        normalizations = (
            RangeNormalization(xl, xu, xl_new, xu_new),
            ZeroToOneNormalization(xl, xu),
        )
        x = np.random.randn(1000)
        x_normalized = (
            normalizations[0].forward(x),
            normalizations[1].forward(x) * (xu_new - xl_new) + xl_new,
        )
        x_denormalized = (
            normalizations[0].backward(x),
            normalizations[1].backward((x - xl_new) / (xu_new - xl_new)),
        )

        np.testing.assert_allclose(x_normalized[0], x_normalized[1])
        np.testing.assert_allclose(x_denormalized[0], x_denormalized[1])

    def test_normalized_problem_wrapper__pareto_set__and__evaluate(self):
        problem = Adjiman()
        normalized_problem = NormalizedProblemWrapper(deepcopy(problem))

        self.assertEqual(problem.xl.shape, normalized_problem.xl.shape)
        self.assertEqual(problem.xu.shape, normalized_problem.xu.shape)

        np.testing.assert_allclose(problem.pareto_set(), [[2, 0.10578]])
        np.testing.assert_allclose(normalized_problem.pareto_set(), [[1, 0.10578]])

        y_opt = problem.pareto_front()
        np.testing.assert_allclose(
            problem.evaluate(problem.pareto_set()), y_opt, atol=1e-4
        )
        np.testing.assert_allclose(
            normalized_problem.evaluate(normalized_problem.pareto_set()),
            y_opt,
            atol=1e-4,
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


if __name__ == "__main__":
    unittest.main()
