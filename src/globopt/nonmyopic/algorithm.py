"""
Implementation of non-myopic version of the Global Optimization strategy based on RBF or
IDW regression from [1]. Here, the algorithm is implemented according to `pymoo` API.

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from typing import Any, Optional

from joblib import Parallel
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.problems.functional import FunctionalProblem
from pymoo.termination.default import DefaultSingleObjectiveTermination

from globopt.myopic.algorithm import GO, Result
from globopt.nonmyopic.acquisition import Array, acquisition
from globopt.util.random import make_seed


class NonMyopicGO(GO):
    """Non-myopic Global Optimization (GO) algorithm based on RBFs and IDWs."""

    def __init__(
        self,
        *args: Any,
        horizon: int,
        discount: float = 1.0,
        shrink_horizon: bool = True,
        rollout_algorithm: Optional[Algorithm] = None,
        acquisition_rollout_kwargs: Optional[dict[str, Any]] = None,
        n_jobs: Optional[int] = -1,
        **kwargs: Any,
    ) -> None:
        """instiantiate a new non-myopic GO algorithm.

        Parameters
        ----------
        horizon : int
            Length of the lookahead predict horizon, which grants non-myopia to the
            algorithm. If `horizon=1`, it falls back to the myopic case.
        discount : float, optional
            Discount factor for the lookahead horizon. By default, `1.0`.
        shrink_horizon : bool, optional
            Whether to shrink the horizon when the algorithm is close to reaching the
            maximum number of iterations, if this limit is specified in the termination.
            By default, `True`.
        rollout_algorithm : Algorithm, optional
            Algorithm to use for the rollout phase of the non-myopic algorithm. By
            default, `PSO` is used.
        acquisition_rollout_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the `minimize` function of the rollout base
            policy.
        n_jobs : int, optional
            Number of parallel jobs to use for the rollout phase. By default, `-1`.
        args, kwargs
            See `globopt.myopic.algorithm.GO`.
        """
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.discount = discount
        self.shrink_horizon = shrink_horizon
        self.rollout_algorithm = rollout_algorithm
        self.acquisition_rollout_kwargs = acquisition_rollout_kwargs or {}
        self.n_jobs = n_jobs

    def _setup(self, problem: Problem, **kwargs: Any) -> None:
        super()._setup(problem, **kwargs)

        if self.rollout_algorithm is None:
            self.rollout_algorithm = PSO()
        if hasattr(self.rollout_algorithm, "pop_size"):
            self.rollout_algorithm.pop_size = (
                self.rollout_algorithm.pop_size * problem.n_var
            )

        if "termination" not in self.acquisition_rollout_kwargs:
            self.acquisition_rollout_kwargs[
                "termination"
            ] = DefaultSingleObjectiveTermination(ftol=1e-4, n_max_gen=300, period=10)
        if seed := kwargs.get("seed", None):
            self.acquisition_rollout_kwargs["seed"] = make_seed(
                str(seed + self.horizon + int(self.discount * 100))
            )

        self.parallel = Parallel(self.n_jobs, verbose=0)
        self.parallel.__enter__()

    def _finalize(self):
        self.parallel.__exit__(None, None, None)

    def _get_acquisition_problem(self) -> Problem:
        # shrink horizon if needed
        h = self.horizon
        if self.shrink_horizon and hasattr(self.termination, "n_max_gen"):
            h = self.horizon = min(h, self.termination.n_max_gen - self.n_iter + 1)

        # define acquisition function problem
        problem: Problem = self.problem
        return FunctionalProblem(
            problem.n_var,
            lambda x: acquisition(
                x,
                self.regression,
                h,
                self.discount,
                self.c1,
                self.c2,
                self.rollout_algorithm,
                problem.xl,
                problem.xu,
                self.parallel,
                **self.acquisition_rollout_kwargs,
            ),
            xl=problem.xl,
            xu=problem.xu,
            elementwise=False,  # enables vectorized evaluation of acquisition function
        )

    def _get_new_sample_from_acquisition_result(self, result: Result) -> Array:
        # from the optimal trajectory, take only the first point
        n_var = self.problem.n_var
        return result.X[:n_var].reshape(1, n_var)

    def __getstate__(self) -> dict[str, Any]:
        # joblib.Parallel cannot be deepcopied
        state = self.__dict__.copy()
        state.pop("parallel", None)
        return state
