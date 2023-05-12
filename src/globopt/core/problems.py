"""
Collection of popular tests for benchmarking optimization algorithms. These tests were
implemented according to [1, 2].

References
----------
[1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
    optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
[2] Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments: Test
    Functions and Datasets. Retrieved May 3, 2023, from
    http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano.
"""


from numbers import Number
from typing import Any, Literal, Type, Union

import numpy as np
import numpy.typing as npt
import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem
from pymoo.problems.single import Ackley as Ackley_original
from pymoo.problems.single import Himmelblau as Himmelblau_original
from pymoo.problems.single import Rosenbrock as Rosenbrock_original

from globopt.core.regression import Array
from globopt.util.normalization import NormalizedProblemWrapper

# in some problems do not return 0 as pareto front, as this produces a bug in which
# f_gap is not computed
ALMOSTZERO = float(np.finfo(np.float64).tiny)


class Ackley(Ackley_original):
    """Ackley benchmark function as per [1] (fix for optimal point and upper and lower
    bound specification).

    References
    ----------
    [1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
        optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
    """

    def __init__(
        self,
        n_var: int = 2,
        a: float = 20,
        b: float = 1 / 5,
        c: float = 2 * np.pi,
        xl: float = -5,
        xu: float = +5,
    ) -> None:
        super().__init__(n_var, a, b, c)
        self.xl = np.full(n_var, xl)
        self.xu = np.full(n_var, xu)

    def _calc_pareto_front(self) -> float:
        return ALMOSTZERO


class Adjiman(Problem):
    """Adjiman benchmark function as per [1].

    References
    ----------
    [1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
        optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
    """

    def __init__(
        self,
        xl: Union[Number, npt.ArrayLike] = (-1, -1),
        xu: Union[Number, npt.ArrayLike] = (2, 1),
    ) -> None:
        super().__init__(n_var=2, n_obj=1, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x: Array, out: dict, *_, **__) -> None:
        x1 = x[:, 0]
        x2 = x[:, 1]
        out["F"] = anp.cos(x1) * anp.sin(x2) - x1 / (anp.square(x2) + 1)

    def _calc_pareto_front(self) -> float:
        return -2.02181

    def _calc_pareto_set(self) -> Array:
        return np.asarray([2, 0.10578])


class Branin(Problem):
    """Branin benchmark function as per [1, 2].

    References
    ----------
    [1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
        optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
    [2] Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments:
        Test Functions and Datasets. Retrieved May 3, 2023, from
        http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano.
    """

    def __init__(
        self,
        a: float = 1,
        b: float = 5.1 / (4 * np.pi**2),
        c: float = 5 / np.pi,
        r: float = 6,
        s: float = 10,
        t: float = 1 / (8 * np.pi),
        xl: Union[Number, npt.ArrayLike] = (-5, 0),
        xu: Union[Number, npt.ArrayLike] = (10, 15),
    ) -> None:
        super().__init__(n_var=2, n_obj=1, xl=xl, xu=xu, vtype=float)
        self.a, self.b, self.c, self.r, self.s, self.t = a, b, c, r, s, t

    def _evaluate(self, x: Array, out: dict, *_, **__) -> None:
        x1 = x[:, 0]
        x2 = x[:, 1]
        out["F"] = (
            self.a * anp.square(x2 - self.b * anp.square(x1) + self.c * x1 - self.r)
            + self.s * (1 - self.t) * anp.cos(x1)
            + self.s
        )

    def _calc_pareto_front(self) -> float:
        return 0.3978873

    def _calc_pareto_set(self) -> Array:
        return np.asarray(
            [
                [-np.pi, 12.275],
                [np.pi, 2.275],
                [3 * np.pi, 2.475],
            ]
        )


class CamelSixHumps(Problem):
    """Six-Humps Camel benchmark function as per [1, 2].

    References
    ----------
    [1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
        optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
    [2] Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments:
        Test Functions and Datasets. Retrieved May 3, 2023, from
        http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano.
    """

    def __init__(
        self,
        xl: Union[Number, npt.ArrayLike] = -5,
        xu: Union[Number, npt.ArrayLike] = 5,
    ) -> None:
        super().__init__(n_var=2, n_obj=1, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x: Array, out: dict, *_, **__) -> None:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1_2 = anp.square(x1)
        x2_2 = anp.square(x2)
        out["F"] = (
            (4 - 2.1 * x1_2 + anp.square(x1_2) / 3) * x1_2
            + x1 * x2
            + (4 * x2_2 - 4) * x2_2
        )

    def _calc_pareto_front(self) -> float:
        return -1.031628453489877

    def _calc_pareto_set(self) -> Array:
        return np.asarray(
            [
                [-0.08984201368301331, 0.7126564032704135],
                [0.08984201368301331, -0.7126564032704135],
            ]
        )


class Hartman(Problem):
    """Hartman benchmark function as per [1, 2].

    References
    ----------
    [1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
        optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
    [2] Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments:
        Test Functions and Datasets. Retrieved May 3, 2023, from
        http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano.
    """

    def __init__(
        self,
        n_var: Literal[3, 6],
        xl: Union[Number, npt.ArrayLike] = 0,
        xu: Union[Number, npt.ArrayLike] = 1,
    ) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu, vtype=float)
        if n_var == 3:
            A = np.asarray([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
            P = np.asarray(
                [
                    [0.36890, 0.1170, 0.2673],
                    [0.46990, 0.4387, 0.7470],
                    [0.10910, 0.8732, 0.5547],
                    [0.03815, 0.5743, 0.8828],
                ]
            )
        elif n_var == 6:
            A = np.asarray(
                [
                    [10, 3, 17, 3.5, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14],
                ]
            )
            P = np.asarray(
                [
                    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
                ]
            )
        else:
            raise ValueError("`n_var` must be either 3 or 6.")
        self.A = A
        self.c = np.asarray([1, 1.2, 3, 3.2])
        self.P = P

    def _evaluate(self, x: Array, out: dict, *_, **__) -> None:
        exponent = -(self.A * np.square(x[:, None, :] - self.P)).sum(axis=-1)
        out["F"] = -(self.c * anp.exp(exponent)).sum(axis=-1)

    def _calc_pareto_front(self) -> float:
        return -3.86278214782076 if self.n_var == 3 else -3.32236801141551

    def _calc_pareto_set(self) -> Array:
        return np.asarray(
            [0.1140, 0.556, 0.852]
            if self.n_var == 3
            else [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301]
        )


class Hartman3(Hartman):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(3, *args, **kwargs)


class Hartman6(Hartman):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(6, *args, **kwargs)


class Himmelblau(Himmelblau_original):
    """Himmelblau benchmark function as per [1] (fix for optimal points).

    References
    ----------
    [1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
        optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
    """

    def _calc_pareto_front(self) -> float:
        return ALMOSTZERO

    def _calc_pareto_set(self) -> Array:
        return np.asarray(
            [
                [3, 2],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )


class Rosenbrock(Rosenbrock_original):
    """Rosenbrock benchmark function as per [1, 2] (fix for optimal points and better
    evaluations and upper and lower bound specifications).

    References
    ----------
    [1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
        optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
    [2] Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments:
        Test Functions and Datasets. Retrieved May 3, 2023, from
        http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano.
    """

    def __init__(
        self,
        n_var: int,
        xl: Union[Number, npt.ArrayLike] = -30,
        xu: Union[Number, npt.ArrayLike] = 30,
    ) -> None:
        super().__init__(n_var)
        self.xl = np.full(n_var, xl)
        self.xu = np.full(n_var, xu)

    def _evaluate(self, x: Array, out: dict, *_, **__) -> None:
        x_i = x[:, :-1]
        x_i_1 = x[:, 1:]
        out["F"] = (
            100 * anp.square(x_i_1 - anp.square(x_i)) + anp.square(1 - x_i)
        ).sum(axis=-1)

    def _calc_pareto_front(self) -> float:
        return ALMOSTZERO

    def _calc_pareto_set(self) -> Array:
        return np.full(self.n_var, 1.0)


class Rosenbrock8(Rosenbrock):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(8, *args, **kwargs)


class Step2Function(Problem):
    """Step 2 Function benchmark function as per [1].

    References
    ----------
    [1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
        optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
    """

    def __init__(
        self,
        n_var: int,
        xl: Union[Number, npt.ArrayLike] = -3,
        xu: Union[Number, npt.ArrayLike] = 3,
    ) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x: Array, out: dict, *_, **__) -> None:
        out["F"] = np.square(np.floor(x + 0.5)).sum(axis=-1)

    def _calc_pareto_front(self) -> float:
        return ALMOSTZERO

    def _calc_pareto_set(self) -> Array:
        return np.zeros(self.n_var)


class Step2Function5(Step2Function):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(5, *args, **kwargs)


class StyblinskiTang(Problem):
    """Styblinski-Tang benchmark function as per [1, 2].

    References
    ----------
    [1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
        optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
    [2] Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments:
        Test Functions and Datasets. Retrieved May 3, 2023, from
        http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano.
    """

    def __init__(
        self,
        n_var: int,
        xl: Union[Number, npt.ArrayLike] = -5,
        xu: Union[Number, npt.ArrayLike] = 5,
    ) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x: Array, out: dict, *_, **__) -> None:
        x_2 = anp.square(x)
        x_4 = anp.square(x_2)
        out["F"] = 0.5 * (x_4 - 16 * x_2 + 5 * x).sum(axis=-1)

    def _calc_pareto_front(self) -> float:
        return -39.16599 * self.n_var

    def _calc_pareto_set(self) -> Array:
        return np.full(self.n_var, -2.903534)


class StyblinskiTang5(StyblinskiTang):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(5, *args, **kwargs)


class Simple1DProblem(Problem):
    """Simple scalar problem to be minimized. Taken from [1].

    References
    ----------
    [1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
        functions. Computational Optimization and Applications, 77(2):571–595, 2020
    """

    def __init__(self) -> None:
        super().__init__(n_var=1, n_obj=1, xl=-3, xu=+3, vtype=float)

    def _evaluate(self, x: Array, out: dict[str, Any], *_, **__) -> None:
        out["F"] = self.f(x)

    def _calc_pareto_front(self) -> float:
        return 0.279504

    def _calc_pareto_set(self) -> float:
        return -0.959769

    @staticmethod
    def f(x: Array) -> Array:
        return (
            (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
            + x**2 / 12
            + x / 10
        )


class AnotherSimple1DProblem(Problem):
    """Another Simple scalar problem to be minimized."""

    def __init__(self) -> None:
        super().__init__(n_var=1, n_obj=1, xl=0, xu=1, vtype=float)

    def _evaluate(self, x: Array, out: dict[str, Any], *_, **__) -> None:
        out["F"] = self.f(x)

    def _calc_pareto_front(self) -> float:
        return -0.669169468

    def _calc_pareto_set(self) -> float:
        return 0.328325636

    @staticmethod
    def f(x: Array) -> Array:
        return x + np.sin(4.5 * np.pi * x)


TESTS: dict[str, tuple[Type[Problem], int, Literal["rbf", "idw"]]] = {
    cls.__name__.lower(): (cls, max_evals, regressor_type)  # type: ignore[misc]
    for cls, max_evals, regressor_type in [
        (Ackley, 50, "rbf"),
        (Adjiman, 10, "rbf"),
        (Branin, 40, "rbf"),
        (CamelSixHumps, 10, "rbf"),
        (Hartman3, 50, "rbf"),
        (Hartman6, 100, "rbf"),
        (Himmelblau, 25, "rbf"),
        (Rosenbrock8, 60, "idw"),
        (Step2Function5, 40, "idw"),
        (StyblinskiTang5, 60, "rbf"),
        (Simple1DProblem, 20, "rbf"),
        (AnotherSimple1DProblem, 20, "idw"),
    ]
}


def get_available_benchmark_problems() -> list[str]:
    """Gets the names of all the available benchmark test problems.

    Returns
    -------
    list of str
        Names of all the available benchmark tests.
    """
    return list(TESTS.keys() - get_available_simple_problems())


def get_available_simple_problems() -> list[str]:
    """Gets the names of all the simple test problems.

    Returns
    -------
    list of str
        Names of all the available simpler tests.
    """
    return ["simple1dproblem", "anothersimple1dproblem"]


def get_benchmark_problem(
    name: str, *args: Any, normalize: bool = False, **kwargs: Any
) -> tuple[Problem, int, Literal["rbf", "idw"]]:
    """Gets an instance of a benchmark test problem.

    Parameters
    ----------
    name : str
        Name of the benchmark test.
    args and kwargs
        Arguments and keyword arguments to be passed to the benchmark test constructor.
    normalize : bool, optional
        If `True`, the problem is wrapped in a `NormalizedProblemWrapper` instance.
        Otherwise, the original problem is returned.

    Returns
    -------
    tuple of (Problem, int, str)
        The problem, the maximum number of evaluations and the regression type suggested
        for its optimization.

    Raises
    ------
    KeyError
        Raised if the name of the benchmark test is not found.
    """
    problem_type, max_evals, regressor = TESTS[name.lower()]
    problem = problem_type(*args, **kwargs)
    if normalize:
        problem = NormalizedProblemWrapper(problem)
    return problem, max_evals, regressor


# import matplotlib.pyplot as plt

# problem = Rosenbrock(n_var=2)
# n = 1000
# ls = np.linspace(*problem.bounds(), n)
# X_ = np.stack(np.meshgrid(*ls.T))
# Z_ = problem.evaluate(X_.reshape(problem.n_var, -1).T).reshape(n, n)

# _, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X_[0], X_[1], Z_, cmap="jet")

# # _, ax = plt.subplots()
# # ax.contour(X_[0], X_[1], Z_, levels=50)

# plt.show()
# quit()
