"""Base class for all acquisition-based Global Optimization algorithms."""

from copy import deepcopy
from typing import Any, Optional

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.optimize import minimize

from globopt.core.display import (
    GlobalOptimizationDisplay,
    GlobalOptimizationOutput,
    Output,
)
from globopt.core.regression import Array


class GOBaseAlgorithm(Algorithm):
    def __init__(
        self,
        acquisition_min_algorithm: Algorithm = None,
        acquisition_min_kwargs: Optional[dict[str, Any]] = None,
        output: Output = None,
        **kwargs,
    ) -> None:
        """instiantiate the base GO algorithm.

        Parameters
        ----------
        acquisition_min_algorithm : Algorithm, optional
            Algorithm used to minimize the acquisition function. By default, `PSO`.
        acquisition_min_kwargs : any, optional
            Additional keyword arguments passed to the acquisition min. algorithm.
        output : Output, optional
            Output display of the algorithm to be printed at each iteration if
            `verbose=True`. By default, `GlobalOptimizationOutput`.
        """
        self.acquisition_min_algorithm = acquisition_min_algorithm
        self.acquisition_min_kwargs = acquisition_min_kwargs or {}
        super().__init__(output=output or GlobalOptimizationOutput(), **kwargs)

    def _setup(self, problem: Problem, **kwargs: Any) -> None:
        """Makes sure the algorithm is set up correctly (modify kwargs and population
        size for the acquisition minimizer, modify the display, etc.)"""
        if self.acquisition_min_algorithm is None:
            self.acquisition_min_algorithm = PSO()

        self.acquisition_min_kwargs["seed"] = None  # would set seed globally (bad)
        self.acquisition_min_kwargs["copy_algorithm"] = True

        verbose = kwargs.get("verbose", False)
        progress = kwargs.get("progress", False)
        acq_output = self.acquisition_min_algorithm.output
        if not verbose:
            self.acquisition_min_kwargs["display"] = None
            self.acquisition_min_kwargs["verbose"] = False
        elif not self.acquisition_min_kwargs.get("verbose", False):
            self.acquisition_min_kwargs["display"] = None
        else:
            self.display = GlobalOptimizationDisplay(self.output, progress, verbose)
            self.acquisition_min_kwargs["display"] = GlobalOptimizationDisplay(
                acq_output,
                progress,
                verbose,
                force_header=False,
                header_prefix="\n\tINNER ITERATIONS:\n",
                line_prefix="\t",
            )
        self._acq_output = acq_output

    def _infill(self) -> Population:
        """Creates one offspring (new point to evaluate) by minimizing acquisition."""
        if acq_min_disaply := self.acquisition_min_kwargs["display"]:
            acq_min_disaply.output = deepcopy(self._acq_output)

        acq_problem = self._get_acquisition_problem()
        res = minimize(
            acq_problem,
            self.acquisition_min_algorithm,
            **self.acquisition_min_kwargs,
        )
        self.acquisition_min_res = res  # for logging purposes
        xnew = self._get_new_sample_from_acquisition_result(res)
        return Population.merge(self.pop, Population.new(X=xnew))

    def _get_acquisition_problem(self) -> Problem:
        """Internal utility to define the acquisition function problem."""
        raise NotImplementedError

    def _get_new_sample_from_acquisition_result(self, result: Result) -> Array:
        """Returns the new point to sample from the acquisition function result."""
        raise NotImplementedError
