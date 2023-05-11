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
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.util.display.output import Output

from globopt.core.regression import DELTA, Idw, Rbf, fit, partial_fit
from globopt.myopic.acquisition import acquisition
from globopt.util.output import GlobalOptimizationOutput, PrefixedStream


class GO(Algorithm):
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
        if acquisition_min_algorithm is None:
            acquisition_min_algorithm = PSO()
        self.acquisition_min_algorithm = acquisition_min_algorithm
        self.acquisition_min_kwargs = acquisition_min_kwargs or {}
        self.acquisition_fun_kwargs = acquisition_fun_kwargs or {}
        super().__init__(output=output or GlobalOptimizationOutput(), **kwargs)

    def _setup(self, problem: Problem, **kwargs: Any) -> None:
        """Makes sure the algorithm is set up correctly."""
        super()._setup(problem, **kwargs)
        self.acquisition_min_kwargs["seed"] = None  # would set seed globally (bad)
        self.acquisition_min_kwargs["copy_algorithm"] = True
        self.acquisition_min_kwargs["verbose"] = False  # self.verbose
        if hasattr(self.acquisition_min_algorithm, "pop_size"):
            self.acquisition_min_algorithm.pop_size = (
                self.acquisition_min_algorithm.pop_size * problem.n_var
            )

    def _initialize_infill(self) -> None:
        """Initialize population (by sampling, if not provided)."""
        super()._initialize_infill()
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
        super()._initialize_advance(infills, **kwargs)
        X, y = infills.get("X", "F")
        self.regression = fit(self.regression, X[np.newaxis], y[np.newaxis])

    def _infill(self) -> Population:
        """Creates one offspring (new point to evaluate) by minimizing acquisition."""
        super()._infill()

        # solve the acquisition minimization problem
        acq_problem = self._get_acquisition_problem()
        with PrefixedStream.prefixed_print("- - - - "):
            res = minimize(
                acq_problem,
                self.acquisition_min_algorithm,
                **self.acquisition_min_kwargs,
            )
        self.acquisition_min_res = res

        # return population with the new point to evaluate merged as last
        xnew = res.X.reshape(1, acq_problem.n_var)
        return Population.merge(self.pop, Population.new(X=xnew))

    def _advance(self, infills: Population, **kwargs: Any) -> Optional[bool]:
        """Adds new offspring to the regression model."""
        super()._advance(infills, **kwargs)
        self.pop = infills
        Xnew = infills[-1].X.reshape(1, 1, -1)
        ynew = infills[-1].F.reshape(1, 1, -1)
        self.regression = partial_fit(self.regression, Xnew, ynew)
        return True  # succesful iteration: True or None; failed iteration: False

    def _get_acquisition_problem(self) -> FunctionalProblem:
        """Internal utility to define the acquisition function problem."""
        problem: Problem = self.problem  # type: ignore[annotation-unchecked]
        mdl = self.regression
        kwargs = self.acquisition_fun_kwargs
        dym = mdl.ym_.max() - mdl.ym_.min()  # span of observations
        return FunctionalProblem(
            n_var=problem.n_var,
            objs=lambda x: acquisition(x[np.newaxis], mdl, None, dym, **kwargs)[0],
            xl=problem.xl,
            xu=problem.xu,
            elementwise=False,  # enables vectorized evaluation of acquisition function
        )
