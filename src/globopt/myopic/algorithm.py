"""
Implementation of myopic Global Optimization strategy based on RBF or IDW regression.
The scheme was first proposed in [1]. Here, the algorithm is implemented according to
`pymoo` API.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from pymoo.core.algorithm import Algorithm
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.problems.functional import FunctionalProblem
from pymoo.util.display.output import Output

from globopt.core.algorithm_base import Array, GOBaseAlgorithm, Result
from globopt.core.regression import DELTA, Idw, Rbf, fit, partial_fit
from globopt.myopic.acquisition import acquisition


class GO(GOBaseAlgorithm):
    """Myopic Global Optimization (GO) algorithm based on RBFs and IDWs."""

    def __init__(
        self,
        regression: Union[None, Idw, Rbf] = None,
        sampling: Sampling = None,
        init_points: Union[int, npt.ArrayLike] = 5,
        acquisition_min_algorithm: Algorithm = None,
        acquisition_min_kwargs: Optional[dict[str, Any]] = None,
        acquisition_fun_kwargs: Optional[dict[str, Any]] = None,
        output: Output = None,
        **kwargs,
    ) -> None:
        """instiantiate a new GO algorithm.

        Parameters
        ----------
        regression : Rbf or Idw, optional
            The regression model to use. If not specified, `Idw` is used.
        sampling : Sampling, optional
            Sampling strategy to draw initial points, in case `init_points` is an
            integer. By default, `LatinHypercubeSampling` is used.
        init_points : int or array-like, optional
            Either the number of initial points to sample, or the initial points
            themselves.
        acquisition_min_algorithm : Algorithm, optional
            Algorithm used to minimize the acquisition function. By default, `PSO`.
        acquisition_min_kwargs : any, optional
            Additional keyword arguments passed to the acquisition min. algorithm.
        acquisition_fun_kwargs : any, optional
            Additional keyword arguments passed to the acquisition function evaluations.
        output : Output, optional
            Output display of the algorithm to be printed at each iteration if
            `verbose=True`. By default, `GlobalOptimizationOutput`.
        """
        if regression is None:
            regression = Idw()
        self.regression = regression
        if sampling is None:
            sampling = LatinHypercubeSampling()
        self.sampling = sampling
        self.init_points = init_points
        super().__init__(
            acquisition_min_algorithm,
            acquisition_min_kwargs,
            acquisition_fun_kwargs,
            output,
            **kwargs,
        )

    def _setup(self, problem: Problem, **kwargs: Any) -> None:
        super()._setup(problem, **kwargs)
        if hasattr(self.acquisition_min_algorithm, "pop_size"):
            self.acquisition_min_algorithm.pop_size = (
                self.acquisition_min_algorithm.pop_size * problem.n_var
            )

    def _initialize_infill(self) -> None:
        """Initialize population (by sampling, if not provided)."""
        problem: Problem = self.problem  # type: ignore[annotation-unchecked]
        init_points = self.init_points

        # if an integer is provided, then the initial data has to be sampled
        if isinstance(init_points, int):
            initialization = Initialization(
                self.sampling,
                eliminate_duplicates=DefaultDuplicateElimination(epsilon=DELTA),
            )
            return initialization.do(problem, init_points)

        # if initial data is provided, then it has to be a tuple of X and F
        return Population.new(X=np.reshape(init_points, (-1, problem.n_var)))

    def _initialize_advance(self, infills: Population, **kwargs) -> None:
        """Fits the regression model to initial data."""
        X, y = infills.get("X", "F")
        self.regression = fit(self.regression, X, y.reshape(-1))

    def _get_acquisition_problem(self) -> Problem:
        problem: Problem = self.problem
        mdl = self.regression
        dym = mdl.ym_.ptp()
        return FunctionalProblem(
            problem.n_var,
            lambda x: acquisition(x, mdl, None, dym, **self.acquisition_fun_kwargs),
            xl=problem.xl,
            xu=problem.xu,
            elementwise=False,  # enables vectorized evaluation of acquisition function
        )

    def _get_new_sample_from_acquisition_result(self, result: Result) -> Array:
        return result.X.reshape(1, self.problem.n_var)

    def _advance(self, infills: Population, **kwargs: Any) -> Optional[bool]:
        """Adds new offspring to the regression model."""
        super()._advance(infills, **kwargs)
        self.pop = infills
        Xnew = infills[-1].X.reshape(1, -1)
        ynew = infills[-1].F
        self.regression = partial_fit(self.regression, Xnew, ynew)
        return True  # succesful iteration: True or None; failed iteration: False
