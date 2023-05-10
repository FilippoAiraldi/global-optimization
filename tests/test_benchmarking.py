import unittest

import numpy as np
from parameterized import parameterized

from globopt.benchmarking.problems import (
    get_available_benchmark_problems,
    get_benchmark_problem,
)


class TestBenchmark(unittest.TestCase):
    @parameterized.expand((False, True))
    def test__pareto_set_and_front(self, normalized: bool):
        testnames = get_available_benchmark_problems()
        for name in testnames:
            pbl, _ = get_benchmark_problem(name, normalize=normalized)
            ps = pbl.pareto_set()
            pf = pbl.pareto_front()
            pf_actual = pbl.evaluate(ps)
            np.testing.assert_allclose(
                pf_actual.squeeze(), pf.squeeze(), rtol=1e-5, atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
