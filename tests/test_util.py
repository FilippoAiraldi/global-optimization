import sys
import unittest
from copy import deepcopy
from io import StringIO

import numpy as np
from pymoo.util.normalization import ZeroToOneNormalization

from globopt.util.benchmark import Adjiman
from globopt.util.normalization import NormalizedProblemWrapper, RangeNormalization
from globopt.util.output import PrefixedStream


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
