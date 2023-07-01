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


from functools import partial
from typing import Callable, Literal, Union

import numpy as np
from vpso.typing import Array1d, Array2d

# Hartmann problem's constants
_C = np.asarray([1, 1.2, 3, 3.2])
_A3 = np.asarray([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
_P3 = np.asarray(
    [
        [0.36890, 0.1170, 0.2673],
        [0.46990, 0.4387, 0.7470],
        [0.10910, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
)
_A6 = np.asarray(
    [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
)
_P6 = np.asarray(
    [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
)


class Problem:
    """Class repreenting a benchmarking problem."""

    __slots__ = ("f", "dim", "lb", "ub", "x_opt", "f_opt")

    def __init__(
        self,
        f: Callable[[Array2d], Array1d],
        dim: int,
        lb: Union[float, Array1d],
        ub: Union[float, Array1d],
        x_opt: Array2d,
    ) -> None:
        """Creates a new benchmarking problem.

        Parameters
        ----------
        f : Callable[[Array2d], Array1d]
            A function that, given a 2d array of shape `(n_points, dim)`, returns a 1d
            evaluation array of shape `(n_points,)`.
        dim : int
            Dimensionality of the problem.
        lb, ub : float or 1d array
            Lower and upper bounds of the problem.
        x_opt : 2d array
            Minimizer(s) of the problem with shape `(n_minimizers, dim)`.
        """
        self.f = f
        self.dim = dim
        self.lb = np.broadcast_to(lb, dim).astype(np.float64, copy=False)
        self.ub = np.broadcast_to(ub, dim).astype(np.float64, copy=False)
        self.x_opt = np.reshape(x_opt, (-1, dim)).astype(np.float64, copy=False)
        self.f_opt = float(f(self.x_opt[0, np.newaxis]))


def _anothersimple1dproblem(x: Array2d) -> Array1d:
    return x + np.sin(4.5 * np.pi * x)


def _simple1dproblem(x: Array2d) -> Array1d:
    return (
        (1 + x * np.sin(2 * x) * np.cos(3 * x) / (1 + x**2)) ** 2
        + x**2 / 12
        + x / 10
    )


def _ackley(x: Array2d) -> Array1d:
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * np.square(x).sum(1)))
        - np.exp(0.5 * np.cos(2 * np.pi * x).sum(1))
        + np.exp(1)
        + 20
    )


def _adjiman(x: Array2d) -> Array1d:
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.cos(x1) * np.sin(x2) - x1 / (np.square(x2) + 1)


def _branin(x: Array2d) -> Array1d:
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (
        np.square(x2 - 5.1 / (4 * np.pi**2) * np.square(x1) + 5 / np.pi * x1 - 6)
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
        + 10
    )


def _camelsixhumps(x: Array2d) -> Array1d:
    x1 = x[:, 0]
    x2 = x[:, 1]
    x1_2 = np.square(x1)
    x2_2 = np.square(x2)
    return (
        (4 - 2.1 * x1_2 + np.square(x1_2) / 3) * x1_2 + x1 * x2 + (4 * x2_2 - 4) * x2_2
    )


def _hartmann(A: Array2d, P: Array2d, C: Array2d, x: Array2d) -> Array1d:
    exponent = -(A * np.square(x[:, np.newaxis, :] - P)).sum(-1)
    return -(C * np.exp(exponent)).sum(-1)


_hartmann3 = partial(_hartmann, _A3, _P3, _C)
_hartmann3.__name__ = "_hartmann3"
_hartmann6 = partial(_hartmann, _A6, _P6, _C)
_hartmann6.__name__ = "_hartmann6"


def _himmelblau(x: Array2d) -> Array1d:
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.square(np.square(x1) + x2 - 11) + np.square(x1 + np.square(x2) - 7)


def _rosenbrock(x: Array2d) -> Array1d:
    x_i = x[:, :-1]
    x_i_1 = x[:, 1:]
    return (100 * np.square(x_i_1 - np.square(x_i)) + np.square(1 - x_i)).sum(1)


def _step2function(x: Array2d) -> Array1d:
    return np.square(np.floor(x + 0.5)).sum(1)


def _styblinskitang(x: Array2d) -> Array1d:
    x_2 = np.square(x)
    return 0.5 * (np.square(x_2) - 16 * x_2 + 5 * x).sum(1)


# simple problems
AnotherSimple1dProblem = Problem(
    _anothersimple1dproblem, 1, 0, 1, 0.328325636  # f_opt: -0.669169468
)
Simple1dProblem = Problem(_simple1dproblem, 1, -3, +3, -0.959769)  # f_opt: 0.279504


# benchmark functions
Ackley = Problem(_ackley, 2, -5, 5, np.zeros(2))  # f_opt: 0
Adjiman = Problem(_adjiman, 2, -1, (2, 1), (2, 0.10578))  # f_opt: -2.02181
Branin = Problem(  # f_opt: 0.3978873
    _branin,
    2,
    (-5, 0),
    (10, 15),
    [[-np.pi, 12.275], [np.pi, 2.275], [3 * np.pi, 2.475]],
)
CamelSixHumps = Problem(  # f_opt: -1.031628453489877
    _camelsixhumps,
    2,
    -5,
    5,
    [[-0.089842013683013, 0.71265640327041], [0.089842013683013, -0.71265640327041]],
)
Hartmann3 = Problem(  # f_opt: -3.86278214782076
    _hartmann3,
    3,
    0,
    1,
    [0.114614, 0.555649, 0.852547],
)
Hartmann6 = Problem(  # f_opt: -3.32236801141551
    _hartmann6,
    6,
    0,
    1,
    [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301],
)
Himmelblau = Problem(  # f_opt: 0
    _himmelblau,
    2,
    -5,
    5,
    [[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]],
)
Rosenbrock8 = Problem(_rosenbrock, 8, -30, 30, np.full(8, 1.0))  # f_opt: 0
Step2Function5 = Problem(_step2function, 5, -3, 3, np.zeros(5))  # f_opt: 0
StyblinskiTang5 = Problem(  # f_opt: -39.16599 * 5
    _styblinskitang, 5, -5, 5, np.full(5, -2.903534)
)


TESTS: dict[str, tuple[Problem, int, Literal["rbf", "idw"]]] = {
    problem.f.__name__[1:]: (problem, max_evals, regressor_type)  # type: ignore[misc]
    for problem, max_evals, regressor_type in [
        (Ackley, 50, "rbf"),
        (Adjiman, 10, "rbf"),
        (Branin, 40, "rbf"),
        (CamelSixHumps, 10, "rbf"),
        (Hartmann3, 50, "rbf"),
        (Hartmann6, 100, "rbf"),
        (Himmelblau, 25, "rbf"),
        (Rosenbrock8, 60, "idw"),
        (Step2Function5, 40, "idw"),
        (StyblinskiTang5, 60, "rbf"),
        (Simple1dProblem, 20, "rbf"),
        (AnotherSimple1dProblem, 20, "idw"),
    ]
}
assert len(TESTS) == 12


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
    return [Simple1dProblem.f.__name__[1:], AnotherSimple1dProblem.f.__name__[1:]]


def get_benchmark_problem(
    name: str, normalize: bool = False
) -> tuple[Problem, int, Literal["rbf", "idw"]]:
    """Gets an instance of a benchmark test problem.

    Parameters
    ----------
    name : str
        Name of the benchmark test.
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
    problem, max_evals, regressor = TESTS[name.lower()]
    if normalize:
        from globopt.util.normalization import normalize_problem

        problem = normalize_problem(problem)
    return problem, max_evals, regressor


# import matplotlib.pyplot as plt

# problem = ...
# n = 1000
# ls = np.linspace(problem.lb, problem.ub, n)
# X_ = np.stack(np.meshgrid(*ls.T))
# Z_ = problem.f(X_.reshape(problem.dim, -1).T).reshape(n, n)

# _, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X_[0], X_[1], Z_, cmap="jet")

# # _, ax = plt.subplots()
# # ax.contour(X_[0], X_[1], Z_, levels=50)

# plt.show()
# quit()
