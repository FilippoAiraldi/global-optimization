"""
Collection of popular tests for benchmarking optimization algorithms. These tests were
implemented according to [1, 2, 3].

References
----------
[1] Jamil, M., Yang, X.-S.: A literature survey of benchmark functions for global
    optimisation problems. Int. J. Math. Model. Numer. Optim. 4(2):150–194 (2013).
[2] Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments: Test
    Functions and Datasets. Retrieved May 3, 2023, from
    http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano.
[3] Jiang, S., Chai, H., Gonzalez, J. and Garnett, R., 2020, November. BINOCULARS for
    efficient, nonmyopic sequential experimental design. In International Conference on
    Machine Learning (pp. 4794-4803). PMLR.
[4] Wang, Z. and Jegelka, S., 2017, July. Max-value entropy search for efficient
    Bayesian optimization. In International Conference on Machine Learning
    (pp. 3627-3635). PMLR.
[5] Eric, B., Freitas, N. and Ghosh, A., 2007. Active preference learning with discrete
    choice data. Advances in neural information processing systems, 20.
"""

from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Union

import numpy as np
import torch
from botorch.test_functions import (
    Ackley,
    Branin,
    DropWave,
    EggHolder,
    Griewank,
    Hartmann,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang,
)
from botorch.test_functions.synthetic import SyntheticTestFunction
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from torch import Tensor


class SimpleProblem(SyntheticTestFunction):
    r"""Simple problem:

        f(x) = (1 + x sin(2x) cos(3x) / (1 + x^2))^2 + x^2 / 12 + x / 10

    x is bounded [-3, +3], and f in has a global minimum at `x_opt = -0.959769`
    with `f_opt = 0.2795`.
    """

    dim = 1
    _optimal_value = 0.279504
    _optimizers = [(-0.959769,)]
    _bounds = [(-3.0, +3.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        X2 = X.square()
        return (
            (1 + X * torch.sin(2 * X) * torch.cos(3 * X) / (1 + X2)).square()
            + X2 / 12
            + X / 10
        )


class Adjiman(SyntheticTestFunction):
    r"""Adjiman function, a 2-dimensional synthetic test function given by:

        f(x) = cos(x) sin(y) - x / (y^2 + 1).

    x is bounded [-1,2], y in [-1,1]. f has a global minimum at
    `x_opt = (2, 0.10578)` with `f_opt = -2.02181`.
    """

    dim = 2
    _optimal_value = -2.02181
    _optimizers = [(2.0, 0.10578)]
    _bounds = [(-1.0, 2.0), (-1.0, 1.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x = X[..., 0]
        y = X[..., 1]
        return x.cos().mul(y.sin()).addcdiv(x, y.square() + 1.0, value=-1)


class Step2(SyntheticTestFunction):
    r"""Step 2 function, an m-dimensional synthetic test function given by:

        f(x) = sum( floor(x + 0.5)^2 ).

    x is bounded [-100,100] in each dimension. f has infinitely many global minima at
    `[-0.5,0.5]`, with `f_opt = 0`.
    """

    _optimal_value = 0.0
    _optimizers = [(0.0, 0.0)]
    _bounds = [(-100.0, 100.0), (-100.0, 100.0)]

    def __init__(self, dim: int, *args: Any, **kwargs: Any) -> None:
        self.dim = dim
        super().__init__(bounds=[self._bounds[0] for _ in range(dim)], *args, **kwargs)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (X + 0.5).floor().square().sum(-1)


class Himmelblau(SyntheticTestFunction):
    r"""Himmelblau function, a 2-dimensional synthetic test function given by:

        f(x) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2.

    x is bounded [-5,5] in each dimension. f has 4 global minima at
    `x_opt = (3, 2), (-2.80511, 3.13131), (-3.77931, -3.28318), (3.58442, -1.84812)`
    with `f_opt = 0`.
    """

    dim = 2
    _optimal_value = 0.0
    _optimizers = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ]
    _bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        return (x1.square() + x2 - 11).square() + (x1 + x2.square() - 7).square()


class Brochu(SyntheticTestFunction):
    r"""Brochu function, a 2-, 4-, or 6-dimensional synthetic test function given by:

        g(x) = sum_i sin(x_i) + x_i / 3 + sin(12 * x_i)
        f2(x) = -max(g(x) - 1, 0)
        f4(x) = -g(x)
        f6(x) = -g(x)

    x is bounded [0,1] in each dimension. f has the following minimizer
    `x_opt_i = 0.6623009251970219` and optimal values  `f_opt2 = -2.662639755973945`,
    `f_opt4 = -7.32527951194789`, and `f_opt6 = -10.987919267921836`.
    """

    def __init__(self, dim: int, *args: Any, **kwargs: Any) -> None:
        if dim not in (2, 4, 6):
            raise ValueError(f"Brochu with dim {dim} not defined")
        self.dim = dim
        self._optimizers = [(0.6623009251970219,) * dim]
        if dim == 2:
            self._optimal_value = -2.662639755973945
        elif dim == 4:
            self._optimal_value = -7.32527951194789
        else:
            self._optimal_value = -10.987919267921836
        super().__init__(bounds=[(0.0, 1.0) for _ in range(dim)], *args, **kwargs)

    def evaluate_true(self, X: Tensor) -> Tensor:
        g = (X.sin() + X / 3 + (12 * X).sin()).sum(dim=-1)
        return -torch.fmax(g - 1.0, torch.scalar_tensor(0.0)) if self.dim == 2 else -g


class GoldsteinPrice(SyntheticTestFunction):
    r"""Goldstein-Price function, a 2-dimensional synthetic test function given by:

        g(x) = 1 + (x1 + x2 + 1)^2 (19 - 14*x1 + 3*x1^2 - 14*x2 + 6*x1*x2 + 3*x2^2)
        p(x) = 30 + (2*x1 - 3*x2)^2 (18 - 32*x1 + 12*x1^2 + 48*x2 - 36*x1*x2 + 27*x2^2)
        f(x) = g(x) p(x).

    x is bounded [-2,2] in each dimension. f has a global minimum at `x_opt = (0, -1)`
    with `f_opt = 3`.
    """

    dim = 2
    _optimal_value = 3.0
    _optimizers = [(0.0, -1.0)]
    _bounds = [(-2.0, 2.0), (-2.0, 2.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        x1sq = x1.square()
        x2sq = x2.square()
        x12 = x1.mul(x2)
        g = 1 + (x1 + x2 + 1).square() * (
            19 - 14 * x1 + 3 * x1sq - 14 * x2 + 6 * x12 + 3 * x2sq
        )
        p = 30 + (2 * x1 - 3 * x2).square() * (
            18 - 32 * x1 + 12 * x1sq + 48 * x2 - 36 * x12 + 27 * x2sq
        )
        return p * g


class Bohachevsky(SyntheticTestFunction):
    r"""Bohachevsky function, a 2-dimensional synthetic test function given by:

        f(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1) - 0.4*cos(4*pi*x2) + 0.7.

    x is bounded [-100,100] in each dimension. f has a global minimum at
    `x_opt = (0, 0)` with `f_opt = 0.0`.
    """

    dim = 2
    _optimal_value = 0.0
    _optimizers = [(0.0, 0.0)]
    _bounds = [(-100.0, 100.0), (-100.0, 100.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        return (
            x1.square()
            + 2 * x2.square()
            - 0.3 * (3 * torch.pi * x1).cos()
            - 0.4 * (4 * torch.pi * x2).cos()
            + 0.7
        )


class Shubert(SyntheticTestFunction):
    r"""Shubert function, a 2-dimensional synthetic test function given by:

        f(x) = prod_i sum_j cos((j + 1) * x_i + j).

    x is bounded [-5.12,5.12] in each dimension. f has 18 global minima at with
    `f_opt = -186.7309`."""

    dim = 2
    _optimal_value = -186.7309
    _optimizers = [
        (-7.0835, 4.858),
        (-7.0835, -7.7083),
        (-1.4251, -7.0835),
        (5.4828, 4.858),
        (-1.4251, -0.8003),
        (4.858, 5.4828),
        (-7.7083, -7.0835),
        (-7.0835, -1.4251),
        (-7.7083, -0.8003),
        (-7.7083, 5.4828),
        (-0.8003, -7.7083),
        (-0.8003, -1.4251),
        (-0.8003, 4.858),
        (-1.4251, 5.4828),
        (5.4828, -7.7083),
        (4.858, -7.0835),
        (5.4828, -1.4251),
        (4.858, -0.8003),
    ]
    _bounds = [(-5.12, 5.12), (-5.12, 5.12)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        ndim = X.ndim - 1
        I = torch.arange(1, 6, dtype=X.dtype, device=X.device).view(5, *(1,) * ndim)
        Ip1 = I + 1
        p1 = torch.cos(Ip1 * X[..., 0].unsqueeze(0) + I).mul(I).sum(dim=0)
        p2 = torch.cos(Ip1 * X[..., 1].unsqueeze(0) + I).mul(I).sum(dim=0)
        return p1 * p2


class Bukin(SyntheticTestFunction):
    r"""Bukin function, a 2-dimensional synthetic test function given by:

        f(x) = 100 * sqrt(abs(x2 - 0.01 * x1^2)) + 0.01 * abs(x1 + 10).

    x is bounded [-15,-5] in the first dimension and [-3,3] in the second dimension.
    f has a global minimum at `x_opt = (-10, 1)` with `f_opt = 0.0`.
    """

    dim = 2
    _optimal_value = 0.0
    _optimizers = [(-10.0, 1.0)]
    _bounds = [(-15.0, -5.0), (-3.0, 3.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        return 100.0 * (x2 - 0.01 * x1.square()).abs().sqrt() + 0.01 * (x1 + 10.0).abs()


class HyperTuningGridTestFunction(SyntheticTestFunction):
    """Test function for hyperparameter tuning. Given a grid of pre-computed points, it
    fits a regressor to interpolate function values at new points.

    Inspired by https://github.com/shalijiang/bo's `hyper_tuning_functions_on_grid.py`.
    """

    def __init__(
        self,
        dataname: str,
        noise_std: Union[None, float, list[float]] = None,
        negate: bool = False,
    ) -> None:
        data = np.genfromtxt(dataname, delimiter=",")
        is_not_nan = np.logical_not(np.any(np.isnan(data), axis=1))
        data = data[is_not_nan, :]
        self.dim = data.shape[1] - 1
        bounds = [(data[:, i].min(), data[:, i].max()) for i in range(self.dim)]

        opt_idx = np.argmin(data[:, -1])
        self._optimal_value = data[opt_idx, -1]
        self._optimizers = [tuple(data[opt_idx, :-1])]

        path = Path(dataname)
        model_path = path.parent / f"{path.stem}.model"
        try:
            self.model = load(model_path)
        except (FileNotFoundError, EOFError):
            self.model = RandomForestRegressor(n_estimators=200)
            self.model.fit(data[:, :-1], data[:, -1])
            dump(self.model, model_path)

        super().__init__(noise_std, negate, bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        with torch.no_grad():
            Y = self.model.predict(X.cpu().numpy())
            return torch.as_tensor(Y, dtype=X.dtype, device=X.device)


class Lda(HyperTuningGridTestFunction):
    """Online Latent Dirichlet allocation (LDA) for Wikipedia articles."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("benchmarking/data/lda_on_grid.csv", *args, **kwargs)


class LogReg(HyperTuningGridTestFunction):
    """Logistic regression for the MNIST dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("benchmarking/data/logreg_on_grid.csv", *args, **kwargs)


class NnBoston(HyperTuningGridTestFunction):
    """Neural network hyperparameter tuning for the Boston housing dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("benchmarking/data/nn_boston_on_grid.csv", *args, **kwargs)


class NnCancer(HyperTuningGridTestFunction):
    """Neural network hyperparameter tuning for the breast cancer dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("benchmarking/data/nn_cancer_on_grid.csv", *args, **kwargs)


class RobotPush3(HyperTuningGridTestFunction):
    """Robot pushing task (3-dimensional)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("benchmarking/data/robotpush3_on_grid.csv", *args, **kwargs)


class RobotPush4(HyperTuningGridTestFunction):
    """Robot pushing task (4-dimensional)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("benchmarking/data/robotpush4_on_grid.csv", *args, **kwargs)


class Svm(HyperTuningGridTestFunction):
    """Structured support vector machine (SVM) on UniPROBE dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("benchmarking/data/svm_on_grid.csv", *args, **kwargs)


Ackley2 = partial(Ackley, dim=2)
Ackley2.__name__ = Ackley.__name__ + "2"
Ackley5 = partial(Ackley, dim=5)
Ackley5.__name__ = Ackley.__name__ + "5"
Brochu2 = partial(Brochu, dim=2)
Brochu2.__name__ = Brochu.__name__ + "2"
Brochu4 = partial(Brochu, dim=4)
Brochu4.__name__ = Brochu.__name__ + "4"
Brochu6 = partial(Brochu, dim=6)
Brochu6.__name__ = Brochu.__name__ + "6"
Hartmann3 = partial(Hartmann, dim=3)
Hartmann3.__name__ = Hartmann.__name__ + "3"
Hartmann6 = partial(Hartmann, dim=6)
Hartmann6.__name__ = Hartmann.__name__ + "6"
Shekel5 = partial(Shekel, m=5)
Shekel5.__name__ = Shekel.__name__ + "5"
Shekel7 = partial(Shekel, m=7)
Shekel7.__name__ = Shekel.__name__ + "7"


TESTS: dict[
    str, tuple[type[SyntheticTestFunction], dict[str, Any], int, Literal["rbf", "idw"]]
] = MappingProxyType(
    {
        problem.__name__.lower(): (problem, kwargs, max_evals, regressor_type)
        for problem, kwargs, max_evals, regressor_type in [
            (Ackley2, {}, 80, "idw"),
            (Ackley5, {}, 50, "idw"),
            (Adjiman, {}, 20, "idw"),
            (Bohachevsky, {}, 25, "idw"),
            (Branin, {}, 40, "idw"),
            (Brochu2, {}, 50, "idw"),
            (Brochu4, {}, 80, "idw"),
            (Brochu6, {}, 80, "idw"),
            (Bukin, {}, 40, "idw"),
            (DropWave, {}, 40, "idw"),
            (EggHolder, {}, 40, "idw"),
            (GoldsteinPrice, {}, 25, "idw"),
            (Griewank, {"dim": 3}, 35, "rbf"),
            (Hartmann3, {}, 50, "rbf"),
            (Hartmann6, {}, 100, "rbf"),
            (Himmelblau, {}, 25, "rbf"),
            (Lda, {}, 30, "idw"),
            (LogReg, {}, 30, "idw"),
            (NnBoston, {}, 100, "rbf"),
            (NnCancer, {}, 80, "rbf"),
            (Rastrigin, {"dim": 4}, 80, "rbf"),
            (RobotPush3, {}, 90, "idw"),
            (RobotPush4, {}, 100, "rbf"),
            (Rosenbrock, {"dim": 8}, 50, "rbf"),
            (Shekel5, {}, 80, "rbf"),
            (Shekel7, {}, 100, "rbf"),
            (Shubert, {}, 25, "rbf"),
            (SixHumpCamel, {"bounds": [(-5.0, 5.0), (-5.0, 5.0)]}, 10, "rbf"),
            (Step2, {"dim": 5}, 80, "idw"),
            (StyblinskiTang, {"dim": 5}, 60, "rbf"),
            (Svm, {}, 60, "idw"),
        ]
    }
)


def get_available_benchmark_problems() -> list[str]:
    """Gets the names of all the available benchmark test problems.

    Returns
    -------
    list of str
        Names of all the available benchmark tests.
    """
    return list(TESTS.keys())


def get_benchmark_problem(
    name: str,
) -> tuple[SyntheticTestFunction, int, Literal["rbf", "idw"]]:
    """Gets an instance of a benchmark synthetic problem.

    Parameters
    ----------
    name : str
        Name of the benchmark test.

    Returns
    -------
    tuple of (SyntheticTestFunction, int, str)
        The problem, the maximum number of evaluations and the regression type suggested
        for its optimization.

    Raises
    ------
    KeyError
        Raised if the name of the benchmark test is not found.
    """
    cls, kwargs, max_evals, regressor = TESTS[name.lower()]
    return cls(**kwargs), max_evals, regressor
