import unittest

import torch
from parameterized import parameterized

from globopt.problems import (
    Ackley2,
    Ackley5,
    Adjiman,
    Bohachevsky,
    Branin,
    Brochu2,
    Brochu4,
    Brochu6,
    Bukin,
    DropWave,
    EggHolder,
    GoldsteinPrice,
    Griewank,
    Hartmann3,
    Hartmann6,
    Himmelblau,
    HyperTuningGridTestFunction,
    Lda,
    LogReg,
    NnBoston,
    NnCancer,
    Rastrigin,
    RobotPush3,
    RobotPush4,
    Rosenbrock,
    Shekel5,
    Shekel7,
    Shubert,
    SixHumpCamel,
    Step2,
    StyblinskiTang,
    Svm,
    get_available_benchmark_problems,
    get_benchmark_problem,
)

CLS: list[type, float] = [
    (Ackley2, 0.0),
    (Ackley5, 0.0),
    (Adjiman, -2.02181),
    (Bohachevsky, 0.0),
    (Branin, 0.3978873),  # 5
    (Brochu2, -2.662639755973945),
    (Brochu4, -7.32527951194789),
    (Brochu6, -10.987919267921836),
    (Bukin, 0.0),
    (DropWave, -1.0),  # 10
    (EggHolder, -959.6407),
    (GoldsteinPrice, 3.0),
    (Griewank, 0.0),
    (Hartmann3, -3.86278214782076),
    (Hartmann6, -3.32236801141551),  # 15
    (Himmelblau, 0.0),
    (Lda, 1266.17),
    (LogReg, 0.0685),
    (NnBoston, 6.5212),
    (NnCancer, 0.040576),  # 20
    (Rastrigin, 0.0),
    (RobotPush3, 0.074788),
    (RobotPush4, 0.076187),
    (Rosenbrock, 0.0),
    (Shekel5, -10.1532),
    (Shekel7, -10.4029),  # 25
    (Shubert, -186.7309),
    (SixHumpCamel, -1.0316),
    (Step2, 0.0),
    (StyblinskiTang, -39.16599 * 5.0),
    (Svm, 0.2411),  # 30
]
EXPECTED_F_OPT: dict[str, float] = {cls.__name__.lower(): f_opt for cls, f_opt in CLS}


class TestProblems(unittest.TestCase):
    def test_list_of_problems__is_sorted(self):
        problems = get_available_benchmark_problems()
        self.assertListEqual(problems, sorted(problems))

    @parameterized.expand([(cls,) for cls, _ in CLS])
    def test_optimal_value_and_point(self, cls: type):
        name = cls.__name__.lower()
        try:
            problem, _, _ = get_benchmark_problem(name)
        except KeyError:
            problem = cls()
        expected = EXPECTED_F_OPT[name]

        actual = problem._optimal_value
        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-6, msg=f"{name} f_opt"
        )

        if problem._optimizers is not None:
            tol = 1e0 if isinstance(problem, HyperTuningGridTestFunction) else 1e-4
            for i, x_opt in enumerate(problem._optimizers):
                f_computed = problem(torch.as_tensor(x_opt))
                expected_ = torch.as_tensor(expected).view_as(f_computed).to(f_computed)
                torch.testing.assert_close(
                    f_computed, expected_, rtol=tol, atol=tol, msg=f"{name} x_opt {i}"
                )


if __name__ == "__main__":
    unittest.main()
