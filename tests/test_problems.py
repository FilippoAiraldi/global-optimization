import unittest

import torch
from parameterized import parameterized

from globopt.problems import (
    Ackley,
    Adjiman,
    Branin,
    Hartmann,
    Michalewicz,
    Rosenbrock,
    SimpleProblem,
    SixHumpCamel,
    StyblinskiTang,
    get_available_benchmark_problems,
    get_benchmark_problem,
)

CLS: list[type, float] = [
    (SimpleProblem, 0.279504),
    (Ackley, 0.0),
    (Adjiman, -2.02181),
    (Branin, 0.3978873),
    (Hartmann, -3.86278214782076),
    (Michalewicz, -4.687658),
    (Rosenbrock, 0.0),
    (SixHumpCamel, -1.0316),
    (StyblinskiTang, -39.16599 * 5.0),
]
EXPECTED_F_OPT: dict[str, float] = {cls.__name__.lower(): f_opt for cls, f_opt in CLS}


class TestProblems(unittest.TestCase):
    def test_list_of_problems__is_sorted(self):
        problems = get_available_benchmark_problems()
        self.assertListEqual(problems, sorted(problems))

    @parameterized.expand([(cls,) for cls, _ in CLS])
    def test_optimal_value_and_point(self, cls: type):
        cls_name = cls.__name__.lower()
        try:
            problem, _, _ = get_benchmark_problem(cls_name)
        except KeyError:
            problem = cls()
        expected = EXPECTED_F_OPT[cls_name]
        actual = problem._optimal_value
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=cls_name)
        if problem._optimizers is not None:
            for x_opt in problem._optimizers:
                f_computed = problem(torch.as_tensor(x_opt))
                expected_ = torch.as_tensor(expected).view_as(f_computed)
                torch.testing.assert_close(
                    f_computed, expected_, rtol=1e-4, atol=1e-6, msg=cls_name
                )


if __name__ == "__main__":
    unittest.main()
