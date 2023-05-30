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
from pymoo.core.problem import Problem
from pymoo.problems.functional import FunctionalProblem

from globopt.myopic.algorithm import GO, Result
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

    def _internal_setup(self, problem: Problem) -> None:
        if hasattr(self.acquisition_min_algorithm, "pop_size"):
            self.acquisition_min_algorithm.pop_size = (
                self.acquisition_min_algorithm.pop_size * problem.n_var * self.horizon
            )
        self.acquisition_fun_kwargs["discount"] = self.discount

    def _get_acquisition_problem(self) -> Problem:
        # shrink horizon if needed
        h = self.horizon
        if self.shrink_horizon and hasattr(self.termination, "n_max_gen"):
            h = self.horizon = min(h, self.termination.n_max_gen - self.n_iter + 1)

        # define acquisition function problem
        problem: Problem = self.problem
        n_var = problem.n_var

        def obj(x: Array) -> Array:
            # transform x from (n_samples, n_var * h) to (n_samples, h, n_var)
            return acquisition(
                x.reshape(-1, h, n_var), self.regression, **self.acquisition_fun_kwargs
            )

        return FunctionalProblem(
            n_var=n_var * h,
            objs=obj,
            xl=np.tile(problem.xl, h),
            xu=np.tile(problem.xu, h),
            elementwise=False,  # enables vectorized evaluation of acquisition function
        )

    def _get_new_sample_from_acquisition_result(self, result: Result) -> Array:
        # from the optimal trajectory, take only the first point
        n_var = self.problem.n_var
        return result.X[:n_var].reshape(1, n_var)
