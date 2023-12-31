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
"""

from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Union

import numpy as np
import torch
from botorch.test_functions import (
    Ackley,
    Branin,
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

    x is bounded [-1,2], y in [-1,1]. f in has a global minimum at
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
    r"""Step 2 function, a m-dimensional synthetic test function given by:

        f(x) = sum( floor(x + 0.5)^2 ).

    x is bounded [-100,100] in each dimension. f in has infinitely many global minima at
    `[-0.5,0.5]`, with `f_opt = 0`.
    """

    _optimal_value = 0.0
    _optimizers = [(0.0, 0.0)]
    _bounds = [(-100.0, 100.0), (-100.0, 100.0)]

    def __init__(self, dim: int, *args: Any, **kwargs: Any) -> None:
        self.dim = dim
        super().__init__(bounds=[(-100.0, 100.0) for _ in range(dim)], *args, **kwargs)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (X + 0.5).floor().square().sum(-1)


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


TESTS: dict[
    str, tuple[type[SyntheticTestFunction], dict[str, Any], int, Literal["rbf", "idw"]]
] = MappingProxyType(
    {
        problem.__name__.lower(): (problem, kwargs, max_evals, regressor_type)
        for problem, kwargs, max_evals, regressor_type in [
            (Ackley, {}, 50, "idw"),
            (Adjiman, {}, 20, "idw"),
            (Branin, {}, 40, "idw"),
            (Hartmann, {"dim": 3}, 50, "rbf"),
            (Lda, {}, 30, "idw"),
            (LogReg, {}, 30, "idw"),
            (NnBoston, {}, 100, "rbf"),
            (NnCancer, {}, 80, "rbf"),
            (Rastrigin, {"dim": 4}, 80, "rbf"),
            (RobotPush3, {}, 90, "idw"),
            (RobotPush4, {}, 100, "rbf"),
            (Rosenbrock, {"dim": 8}, 50, "rbf"),
            (Shekel, {"m": 7}, 80, "rbf"),
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
