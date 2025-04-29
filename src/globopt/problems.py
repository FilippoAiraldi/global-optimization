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
from importlib import resources
from math import pi, sqrt
from typing import Any, Callable, Literal, Optional, Union
from warnings import warn

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
from botorch.test_functions.synthetic import ConstrainedGramacy as _ConstrainedGramacy
from botorch.test_functions.synthetic import (
    ConstrainedHartmannSmooth as ConstrainedHartmann6,
)
from botorch.test_functions.synthetic import ConstrainedSyntheticTestFunction
from botorch.test_functions.synthetic import PressureVessel as _PressureVessel
from botorch.test_functions.synthetic import SpeedReducer as _SpeedReducer
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.test_functions.synthetic import WeldedBeamSO as _WeldedBeamSO
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from torch import Tensor

from globopt.sampling import latin_hypercube_with_nonlinear_constraint


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
            (1 + X * (2 * X).sin() * (3 * X).cos() / (1 + X2)).square()
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
        return x.cos() * y.sin() - x / (y.square() + 1.0)


class Step2(SyntheticTestFunction):
    r"""Step 2 function, an m-dimensional synthetic test function given by:

        f(x) = sum( floor(x + 0.5)^2 ).

    x is bounded [-100,100] in each dimension. f has infinitely many global minima at
    the origin, with `f_opt = 0`.
    """

    _optimal_value = 0.0
    _optimizers = [(0.0,)]
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
        return (g - 1.0).clamp_min(0.0).neg() if self.dim == 2 else g.neg()


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

    x is bounded [-5.12,5.12] in each dimension. f has 18 global minima at various
    locations with `f_opt = -186.7309`."""

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
        with resources.path("globopt.data", dataname) as datapath:
            data = np.genfromtxt(datapath, delimiter=",")
            modelpath = datapath.with_suffix(".model")

        is_not_nan = np.logical_not(np.any(np.isnan(data), axis=1))
        data = data[is_not_nan, :]
        self.dim = data.shape[1] - 1
        bounds = [(data[:, i].min(), data[:, i].max()) for i in range(self.dim)]

        opt_idx = np.argmin(data[:, -1])
        self._optimal_value = data[opt_idx, -1]
        self._optimizers = [tuple(data[opt_idx, :-1])]

        try:
            self.model = load(modelpath)
        except (FileNotFoundError, EOFError):
            warn(
                f'Preparing a regression model for "{dataname}". This can take some '
                "time",
                UserWarning,
                2,
            )
            self.model = RandomForestRegressor(n_estimators=200)
            self.model.fit(data[:, :-1], data[:, -1])
            dump(self.model, modelpath)

        super().__init__(noise_std, negate, bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        with torch.no_grad():
            Y = self.model.predict(X.cpu().numpy())
            return torch.as_tensor(Y, dtype=X.dtype, device=X.device)


class Lda(HyperTuningGridTestFunction):
    """Online Latent Dirichlet allocation (LDA) for Wikipedia articles."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("lda_on_grid.csv", *args, **kwargs)


class LogReg(HyperTuningGridTestFunction):
    """Logistic regression for the MNIST dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("logreg_on_grid.csv", *args, **kwargs)


class NnBoston(HyperTuningGridTestFunction):
    """Neural network hyperparameter tuning for the Boston housing dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("nn_boston_on_grid.csv", *args, **kwargs)


class NnCancer(HyperTuningGridTestFunction):
    """Neural network hyperparameter tuning for the breast cancer dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("nn_cancer_on_grid.csv", *args, **kwargs)


class RobotPush3(HyperTuningGridTestFunction):
    """Robot pushing task (3-dimensional)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("robotpush3_on_grid.csv", *args, **kwargs)


class RobotPush4(HyperTuningGridTestFunction):
    """Robot pushing task (4-dimensional)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("robotpush4_on_grid.csv", *args, **kwargs)


class Svm(HyperTuningGridTestFunction):
    """Structured support vector machine (SVM) on UniPROBE dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("svm_on_grid.csv", *args, **kwargs)


class Cosmological(HyperTuningGridTestFunction):
    """Estimation of cosmological constants of a physical model of the Universe."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("cosmological_on_grid.csv", *args, **kwargs)


Ackley2 = partial(Ackley, dim=2)
setattr(Ackley2, "__name__", Ackley.__name__ + "2")
Ackley5 = partial(Ackley, dim=5)
setattr(Ackley5, "__name__", Ackley.__name__ + "5")
Brochu2 = partial(Brochu, dim=2)
setattr(Brochu2, "__name__", Brochu.__name__ + "2")
Brochu4 = partial(Brochu, dim=4)
setattr(Brochu4, "__name__", Brochu.__name__ + "4")
Brochu6 = partial(Brochu, dim=6)
setattr(Brochu6, "__name__", Brochu.__name__ + "6")
Hartmann3 = partial(Hartmann, dim=3)
setattr(Hartmann3, "__name__", Hartmann.__name__ + "3")
Hartmann6 = partial(Hartmann, dim=6)
setattr(Hartmann6, "__name__", Hartmann.__name__ + "6")
Shekel5 = partial(Shekel, m=5)
setattr(Shekel5, "__name__", Shekel.__name__ + "5")
Shekel7 = partial(Shekel, m=7)
setattr(Shekel7, "__name__", Shekel.__name__ + "7")


TESTS: dict[
    str, tuple[type[SyntheticTestFunction], dict[str, Any], int, Literal["rbf", "idw"]]
] = {
    problem.__name__.lower(): (problem, kwargs, max_evals, regressor_type)
    for problem, kwargs, max_evals, regressor_type in [
        (Ackley2, {}, 80, "idw"),
        (Ackley5, {}, 80, "idw"),
        (Adjiman, {}, 25, "idw"),
        (Bohachevsky, {}, 50, "idw"),
        (Branin, {}, 35, "idw"),
        (Brochu2, {}, 50, "idw"),
        (Brochu4, {}, 80, "idw"),
        (Brochu6, {}, 80, "idw"),
        (Bukin, {}, 25, "idw"),
        (Cosmological, {}, 80, "idw"),
        (DropWave, {}, 100, "idw"),
        (EggHolder, {}, 80, "idw"),
        (GoldsteinPrice, {}, 50, "idw"),
        (Griewank, {"dim": 3}, 80, "idw"),
        (Hartmann3, {}, 80, "idw"),
        (Hartmann6, {}, 100, "idw"),
        (Himmelblau, {}, 40, "idw"),
        (Lda, {}, 30, "idw"),
        (LogReg, {}, 25, "idw"),
        (NnBoston, {}, 100, "idw"),
        (NnCancer, {}, 60, "idw"),
        (Rastrigin, {"dim": 4}, 100, "idw"),
        (RobotPush3, {}, 90, "idw"),
        (RobotPush4, {}, 100, "idw"),
        (Rosenbrock, {"dim": 8}, 50, "idw"),
        (Shekel5, {}, 80, "idw"),
        (Shekel7, {}, 100, "idw"),
        (Shubert, {}, 50, "idw"),
        (SixHumpCamel, {"bounds": [(-5.0, 5.0), (-5.0, 5.0)]}, 50, "idw"),
        (Step2, {"dim": 5}, 80, "idw"),
        (StyblinskiTang, {"dim": 5}, 100, "idw"),
        (Svm, {}, 20, "idw"),
    ]
}


def get_available_benchmark_problems() -> list[str]:
    """Gets the names of all the available benchmark test problems.

    Returns
    -------
    list of str
        Names of all the available benchmark tests.
    """
    return list(TESTS.keys())


########################################################################################


class ConstrainedSixHumpCamel(SixHumpCamel, ConstrainedSyntheticTestFunction):
    r"""Constraind six hump camel function."""

    dim = SixHumpCamel.dim
    _optimizers = [
        (
            (684452907 - 5000 * sqrt(2889571934)) / 1950978529,
            (215115 * sqrt(2889571934) - 360078529) / 19509785290,
        )
    ]
    _bounds = [(0.2, 0.8), (-0.4, 0.6)]

    def __init__(
        self, *args: Any, dtype: torch.dtype = torch.double, **kwargs: Any
    ) -> None:
        x1, x2 = self._optimizers[0]
        x1_sq = x1**2
        x2_sq = x2**2
        self._optimal_value = (
            (4 - 2.1 * x1_sq + 1 / 3 * (x1_sq * x1_sq)) * x1_sq
            + x1 * x2
            + (-4 + 4 * x2_sq) * x2_sq
        )
        self.A = torch.as_tensor(
            [
                (-1.6295, -1),
                (1, -4.4553),
                (4.3023, 1),
                (5.6905, 12.1374),
                (-17.6198, -1),
            ],
            dtype,
        )
        self.b = torch.as_tensor([-3.0786, -2.7417, 1.4909, -1.0, -32.5198], dtype)
        self.num_constraints = 1 + self.A.shape[0]
        ConstrainedSyntheticTestFunction.__init__(self, *args, **kwargs, dtype=dtype)

    @staticmethod
    def _nonlinear_inequality_constraint0(X: Tensor) -> Tensor:
        x1, x2 = X.unbind(-1)
        return 0.5 - x1.square() - (x2 + 0.1).square()

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        nonlinear_ineq_con = self._nonlinear_inequality_constraint0(X).unsqueeze(-1)
        lin_ineq_cons = (self.A @ X.unsqueeze(-1)).squeeze(-1) - self.b
        return torch.concat((nonlinear_ineq_con, lin_ineq_cons), dim=-1)


class ConstrainedGramacy(_ConstrainedGramacy):
    """Constrained Gramacy test function with tighetened bounds."""

    _bounds = [(0.0, 1.0), (0.1, 1.0)]

    @staticmethod
    def _nonlinear_inequality_constraint0(X: Tensor) -> Tensor:
        x1, x2 = X.unbind(-1)
        return x1 + 2 * x2 + 0.5 * torch.sin(2 * pi * (x1.pow(2) - 2 * x2)) - 1.5

    @staticmethod
    def _nonlinear_inequality_constraint1(X: Tensor) -> Tensor:
        x1, x2 = X.unbind(-1)
        return 1.5 - x1.pow(2) - x2.pow(2)


class PressureVessel(_PressureVessel):
    """Pressure vessel design test function with tighetened bounds."""

    num_constraints = 3
    _optimal_value = 6059.715
    _bounds = [(0.5, 10.0), (0.25, 10.0), (30, 50.0), (150.0, 240.0)]

    def __init__(
        self, *args: Any, dtype: torch.dtype = torch.double, **kwargs: Any
    ) -> None:
        self.A = torch.as_tensor(
            [(1.0, 0.0, -0.0193, 0.0), (0.0, 1.0, -0.00954, 0.0)], dtype=dtype
        )
        self.b = torch.zeros(2, dtype=dtype)
        super().__init__(*args, **kwargs, dtype=dtype)

    @staticmethod
    def _nonlinear_inequality_constraint0(X: Tensor) -> Tensor:
        x3, x4 = X[..., 2], X[..., 3]
        return x3.square() * x4 + 4 / 3 * x3.pow(3) - 1296000 / pi


class WeldedBeam(_WeldedBeamSO):
    """ "Welded beam design test function with tighetened bounds."""

    _optimal_value = 1.728226
    _bounds = [(0.125, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]

    def __init__(
        self, *args: Any, dtype: torch.dtype = torch.double, **kwargs: Any
    ) -> None:
        self.A = torch.as_tensor([(-1.0, 0.0, 0.0, 1.0)], dtype=dtype)
        self.b = torch.zeros(1, dtype=dtype)
        self._P = 6000.0
        self._L = 14.0
        self._E = 30e6
        self._G = 12e6
        self._t_max = 13600.0
        self._s_max = 30000.0
        self._d_max = 0.25
        super().__init__(*args, **kwargs, dtype=dtype)

    def _nonlinear_inequality_constraint0(self, X: Tensor) -> Tensor:
        x1, x2, x3, _ = X.unbind(-1)
        sqrt2 = sqrt(2)
        M = self._P * (self._L + x2 / 2)
        R = (0.25 * (x2.square() + (x1 + x3).square())).sqrt()
        J = 2 * sqrt2 * x1 * x2 * (x2.square() / 12 + 0.25 * (x1 + x3).square())
        t1 = self._P / (sqrt2 * x1 * x2)
        t2 = M * R / J
        return self._t_max - (t1.square() + t1 * t2 * x2 / R + t2.square()).sqrt()

    def _nonlinear_inequality_constraint1(self, X: Tensor) -> Tensor:
        x3, x4 = X[..., 2], X[..., 3]
        s = 6 * self._P * self._L / (x4 * x3.square())
        return self._s_max - s

    def _nonlinear_inequality_constraint2(self, X: Tensor) -> Tensor:
        x1, x2, x3, x4 = X.unbind(-1)
        return 5.0 - 0.10471 * x1.square() - 0.04811 * x3 * x4 * (14.0 + x2)

    def _nonlinear_inequality_constraint3(self, X: Tensor) -> Tensor:
        x3, x4 = X[..., 2], X[..., 3]
        d = 4 * self._P * self._L**3 / (self._E * x3.pow(3) * x4)
        return self._d_max - d

    def _nonlinear_inequality_constraint4(self, X: Tensor) -> Tensor:
        x3, x4 = X[..., 2], X[..., 3]
        E = self._E
        L = self._L
        C = 4.013 * 6 * E / (L**2)
        P_c = C * x3 * x4.pow(3) * (1 - 0.25 * x3 / L * sqrt(E / self._G))
        return P_c - self._P


class SpeedReducer(_SpeedReducer):
    """Speed reducer design test function with tighetened bounds."""

    _optimal_value = 2996.3482
    _bounds = [
        (3.45, 3.6),
        (0.7, 0.75),
        (17.0, 28.0),
        (7.3, 8.3),
        (7.8, 8.3),
        (3.3, 3.9),
        (5.25, 5.5),
    ]

    @staticmethod
    def _nonlinear_inequality_constraint0(X: Tensor) -> Tensor:
        x1, x2, x3 = X[..., 0], X[..., 1], X[..., 2]
        return 1 - 27 / (x1 * x2.square() * x3)

    @staticmethod
    def _nonlinear_inequality_constraint1(X: Tensor) -> Tensor:
        x1, x2, x3 = X[..., 0], X[..., 1], X[..., 2]
        return 1 - 397.5 / (x1 * x2.square() * x3.square())

    @staticmethod
    def _nonlinear_inequality_constraint2(X: Tensor) -> Tensor:
        x2, x3, x4, x6 = X[..., 1], X[..., 2], X[..., 3], X[..., 5]
        return 1 - 1.93 * x4.pow(3) / (x2 * x3 * x6.pow(4))

    @staticmethod
    def _nonlinear_inequality_constraint3(X: Tensor) -> Tensor:
        x2, x3, x5, x7 = X[..., 1], X[..., 2], X[..., 4], X[..., 6]
        return 1 - 1.93 * x5.pow(3) / (x2 * x3 * x7.pow(4))

    @staticmethod
    def _nonlinear_inequality_constraint4(X: Tensor) -> Tensor:
        x2, x3, x4, x6 = X[..., 1], X[..., 2], X[..., 3], X[..., 5]
        return 110 - x6.pow(-3) * ((745 * x4 / (x2 * x3)).square() + 16.9e6).sqrt()

    @staticmethod
    def _nonlinear_inequality_constraint5(X: Tensor) -> Tensor:
        x2, x3, x5, x7 = X[..., 1], X[..., 2], X[..., 4], X[..., 6]
        return 85 - x7.pow(-3) * ((745 * x5 / (x2 * x3)).square() + 157.5e6).sqrt()

    @staticmethod
    def _nonlinear_inequality_constraint6(X: Tensor) -> Tensor:
        x2, x3 = X[..., 1], X[..., 2]
        return 40 - x2 * x3

    @staticmethod
    def _nonlinear_inequality_constraint7(X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        return x1 / x2 - 5

    @staticmethod
    def _nonlinear_inequality_constraint8(X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        return 12 - x1 / x2

    @staticmethod
    def _nonlinear_inequality_constraint9(X: Tensor) -> Tensor:
        x4, x6 = X[..., 3], X[..., 5]
        return 1 - (1.5 * x6 + 1.9) / x4

    @staticmethod
    def _nonlinear_inequality_constraint10(X: Tensor) -> Tensor:
        x5, x7 = X[..., 4], X[..., 6]
        return 1 - (1.1 * x7 + 1.9) / x5


CONSTRAINED_TESTS: dict[
    str, tuple[type[SyntheticTestFunction], dict[str, Any], int, Literal["rbf", "idw"]]
] = {
    problem.__name__.lower(): (problem, kwargs, max_evals, regressor_type)
    for problem, kwargs, max_evals, regressor_type in [
        (ConstrainedSixHumpCamel, {}, 30, "rbf"),
        (ConstrainedGramacy, {}, 30, "rbf"),
        (ConstrainedHartmann6, {}, 30, "rbf"),
        (PressureVessel, {}, 30, "rbf"),
        (WeldedBeam, {}, 30, "rbf"),
        (SpeedReducer, {}, 30, "rbf"),
    ]
}


def get_problem_constraints_and_ic_generator(
    problem: ConstrainedSyntheticTestFunction,
) -> tuple[
    Optional[list[tuple[Tensor, Tensor, float]]],
    Optional[list[tuple[Callable, bool]]],
    Callable[[int, int], Optional[Tensor]],
]:
    """Given a problem, returns the inequality and nonlinear constraints in a form
    amenable to BoTorch's `optimize_acqf`, and, if necessary, a callable to generate
    initial conditions via constrained scrambled Latin hypercube sampling.

    Parameters
    ----------
    problem : ConstrainedSyntheticTestFunction
        The problem to get the constraints from.

    Returns
    -------
    tuple of two lists and callable
         - list of tuples of (indices, coefficients, rhs) for the linear ineq. constr.,
         - list of tuples of (function, intrapoint) for the nonlinear ineq. constraints.
        Any of the two lists may be `None` if the problem does not have that type of
        constraints.
        Finally, a callable is also returned that generates initial conditions for the
        the problem, if necessary. It returns `None` if no nonlinear ineq. constraints
        are present. Otherwise, it accepts the number of samples `num_restarts` and
        `q`-batches and returns a tensor of shape `(num_restarts, q, dim)` of the
        initial conditions.

    Raises
    ------
    ValueError
        If the problem has some unrecognized constraints.
    """
    if getattr(problem, "num_constraints", 0) <= 0:
        return None, None, lambda *_, **__: None

    # get some constants
    bounds = problem.bounds
    lb, ub = bounds
    dim = problem.dim
    dtype = bounds.dtype

    # process each constrained problem individually. For each, we extract the linear and
    # nonlinear inequality constraints in the form that is required by BoTorch's
    # `optimize_acqf` method. We also create, if at least one nonlinear constraint is
    # present, a callable that finds the minimum of all constraints. This will be then
    # used for LHS.
    if hasattr(problem, "A") and hasattr(problem, "b"):
        indices = torch.arange(dim, dtype=torch.long)
        lin_ineq_constrs = [
            (indices, a.to(dtype), b) for a, b in zip(problem.A, problem.b)
        ]
    else:
        lin_ineq_constrs = None

    nonlin_ineq_constrs = []
    i = 0
    while True:
        func = getattr(problem, f"_nonlinear_inequality_constraint{i}", None)
        if func is None:
            break
        nonlin_ineq_constrs.append((func, True))
        i += 1
    if len(nonlin_ineq_constrs) == 0:
        nonlin_ineq_constrs = None

    if lin_ineq_constrs is None and nonlin_ineq_constrs is None:
        raise ValueError(
            f"Problem {problem.__class__.__name__} has unrecognized constraints."
        )

    # if the problem has at least one nonlinear constraint, we need to sample points
    # via a custom LHS method
    if nonlin_ineq_constrs is None:
        return lin_ineq_constrs, None, lambda *_, **__: None

    def constraint_func(X: Tensor) -> Tensor:
        # X \in (num_restart, q, dim), so we take the minimum of all constraints and
        # along the whole (q) trajectory/batch
        return problem.evaluate_slack_true(X).amin((1, 2))

    def sampler(num_restart: int, q: int) -> Tensor:
        samples = latin_hypercube_with_nonlinear_constraint(
            num_restart, q, dim, lb, ub, constraint_func
        ).view(num_restart, q, dim)
        return samples

    return lin_ineq_constrs, nonlin_ineq_constrs, sampler


def get_available_constrained_benchmark_problems() -> list[str]:
    """Gets the names of all the available benchmark constrained test problems.

    Parameters
    ----------
    include_constrained_problems : bool, optional
        If `True`, the function will include constrained problems in the list. Default
        is `False`.

    Returns
    -------
    list of str
        Names of all the available benchmark tests.
    """
    return list(CONSTRAINED_TESTS.keys())


########################################################################################


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
    name_ = name.lower()
    source = TESTS if name_ in TESTS else CONSTRAINED_TESTS
    cls, kwargs, max_evals, regressor = source[name_]
    return cls(**kwargs), max_evals, regressor
