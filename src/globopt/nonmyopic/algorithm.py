"""
Implementation of non-myopic version of the Global Optimization strategy based on RBF or
IDW regression from [1]. Here, the algorithm is implemented according to `pymoo` API.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from typing import Any

import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.problems.functional import FunctionalProblem

from globopt.myopic.algorithm import GO, PrefixedStream, minimize
from globopt.nonmyopic.acquisition import Array, acquisition


class NonMyopicGO(GO):
    """Non-myopic Global Optimization (GO) algorithm based on RBFs and IDWs."""

    def __init__(
        self,
        *args: Any,
        horizon: int,
        shrink_horizon: bool = False,
        discount: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """instiantiate a new non-myopic GO algorithm.

        Parameters
        ----------
        horizon : int
            Length of the lookahead predict horizon, which grants non-myopia to the
            algorithm. If `horizon=1`, it falls back to the myopic case.
        shrink_horizon : bool, optional
            Whether to shrink the horizon when the algorithm is close to reaching the
            maximum number of iterations, if this limit is specified in the termination.
            By default, `False`.
        discount : float, optional
            Discount factor for the lookahead horizon. By default, `1.0`.
        args, kwargs
            See `globopt.myopic.algorithm.GO`.
        """
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.shrink_horizon = shrink_horizon
        self.discount = discount

    def _setup(self, problem: Problem, **kwargs: Any) -> None:
        super()._setup(problem, **kwargs)
        if hasattr(self.acquisition_min_algorithm, "pop_size"):
            self.acquisition_min_algorithm.pop_size = (
                self.acquisition_min_algorithm.pop_size * problem.n_var * self.horizon
            )
        self.acquisition_fun_kwargs["discount"] = self.discount

    def _infill(self) -> Population:
        """Creates one offspring (new point to evaluate) by minimizing acquisition."""

        # shrink horizon if needed
        h = self.horizon
        if self.shrink_horizon and hasattr(self.termination, "n_max_gen"):
            h = self.horizon = min(h, self.termination.n_max_gen - self.n_iter + 1)

        # define acquisition function problem
        problem: Problem = self.problem  # type: ignore[annotation-unchecked]
        n_var = problem.n_var
        mdl = self.regression
        kwargs = self.acquisition_fun_kwargs

        def obj(x: Array) -> Array:
            # transform x_ from (n_samples, n_var * h) to (n_samples, h, n_var)
            x = x.reshape(-1, h, n_var)
            return acquisition(x, mdl, **kwargs)

        acq_problem = FunctionalProblem(
            n_var=n_var * h,
            objs=obj,
            xl=np.tile(problem.xl, h),
            xu=np.tile(problem.xu, h),
            elementwise=False,  # enables vectorized evaluation of acquisition function
        )

        # solve the acquisition minimization problem
        with PrefixedStream.prefixed_print("- - - - "):
            res = minimize(
                acq_problem,
                self.acquisition_min_algorithm,
                **self.acquisition_min_kwargs,
            )
        self.acquisition_min_res = res

        # return population with the first point of the optimal trajectory to evaluate
        # merged as last
        xnew = res.X[:n_var].reshape(1, n_var)
        return Population.merge(self.pop, Population.new(X=xnew))

    # def _post_advance(self):
    #     return super()._post_advance()
