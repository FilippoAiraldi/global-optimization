# """
# Implementation of the non-myopic acquisition function for RBF/IDW-based Global
# Optimization, based on the myopic function in [1].

# References
# ----------
# [1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
#     functions. Computational Optimization and Applications, 77(2):571–595, 2020
# """


# from typing import Any, Optional

# import numpy as np
# import numpy.typing as npt
# from joblib import Parallel, delayed
# from pymoo.algorithms.soo.nonconvex.pso import PSO
# from pymoo.core.algorithm import Algorithm
# from pymoo.optimize import minimize
# from pymoo.problems.functional import FunctionalProblem
# from pymoo.termination.default import DefaultSingleObjectiveTermination

# from globopt.core.regression import Array, RegressorType, partial_fit, predict
# from globopt.myopic.acquisition import acquisition as myopic_acquisition
# from globopt.util.random import make_seeds


# def _rollout(
#     x: Array,
#     y_hat: Array,
#     mdl: RegressorType,
#     horizon: int,
#     discount: float,
#     c1: float,
#     c2: float,
#     algorithm: Algorithm,
#     xl: Optional[npt.ArrayLike],
#     xu: Optional[npt.ArrayLike],
#     seed: Optional[int],
#     **kwargs: Any,
# ) -> float:
#     """Rollouts the base greedy/myopic policy from the given point, using the regression
#     to predict the evolution of the dynamics of the optimization problem."""
#     n_var = x.size
#     mdl = partial_fit(mdl, x.reshape(1, n_var), y_hat.reshape(1))
#     y_min = mdl.ym_.min()
#     y_max = mdl.ym_.max()
#     a = 0.0
#     for h, seed_ in zip(range(1, horizon), make_seeds(seed)):
#         dym = y_max - y_min
#         problem = FunctionalProblem(
#             n_var,
#             lambda x_: myopic_acquisition(x_, mdl, None, dym, c1, c2),
#             xl=xl,
#             xu=xu,
#             elementwise=False,
#         )
#         res = minimize(
#             problem,
#             algorithm,
#             verbose=False,
#             seed=seed_,
#             **kwargs,
#         )
#         a += res.F.item() * discount**h

#         # add new point to the regression model
#         x = res.X.reshape(1, n_var)
#         y_hat = predict(mdl, x)
#         mdl = partial_fit(mdl, x, y_hat)
#         y_min = min(y_min, y_hat)
#         y_max = max(y_max, y_hat)
#     return a


# def acquisition(
#     x: Array,
#     mdl: RegressorType,
#     horizon: int,
#     discount: float = 1.0,
#     c1: float = 1.5078,
#     c2: float = 1.4246,
#     base_algorithm: Optional[Algorithm] = None,
#     xl: Optional[npt.ArrayLike] = None,
#     xu: Optional[npt.ArrayLike] = None,
#     parallel: Optional[Parallel] = None,
#     seed: Optional[int] = None,
#     **minimize_kwargs: Any,
# ) -> Array:
#     """Computes the non-myopic acquisition function for IDW/RBF regression models.

#     Parameters
#     ----------
#     x : array of shape (n_samples, n_var)
#         Array of points for which to compute the acquisition. `n_samples` is the number
#         of target points for which to compute the acquisition, and `n_var` is the number
#         of features/variables of each point.
#     mdl : RegressorType
#         Fitted model to use for computing the acquisition function.
#     horizon : int
#         Length of the lookahead/non-myopic horizon.
#     discount : float, optional
#         Discount factor of the MPD along the horizon. By default, `1.0`.
#     c1 : float, optional
#         Weight of the contribution of the variance function, by default `1.5078`.
#     c2 : float, optional
#         Weight of the contribution of the distance function, by default `1.4246`.
#     base_algorithm : Algorithm, optional
#         Algorithm to use for the rollout policy. By default, `PSO`.
#     xl, xu : array_like of shape (n_var,), optional
#         Lower and upper bounds on each variable to bound the search in the
#         `base_algorithm`.
#     parallel : Parallel, optional
#         Parallel object to use for parallel computation. By default,
#         `Parallel(n_jobs=-1, verbose=0)` is used.
#     seed : int, optional
#         Seed to use for the random number generator of the `base_algorithm`.
#     minimize_kwargs : dict, optional
#         Additional keyword arguments to pass to the `minimize` function of each rollout
#         policy optimization problem.

#     Returns
#     -------
#     array of shape (n_samples,)
#         The non-myopic acquisition function computed for each `x`.
#     """
#     if parallel is None:
#         parallel = Parallel(n_jobs=-1, verbose=0)  # 10 for debugging
#     if base_algorithm is None:
#         base_algorithm = PSO()
#     if "termination" not in minimize_kwargs:
#         minimize_kwargs["termination"] = DefaultSingleObjectiveTermination(
#             ftol=1e-4, n_max_gen=300, period=10
#         )

#     # compute the cost associated to the one-step lookahead
#     y_hat = predict(mdl, x)
#     a = myopic_acquisition(x, mdl, y_hat, None, c1, c2)
#     if horizon == 1 or discount <= 0.0:
#         return a

#     # for each sample, compute the rollout policy by rolling out the base myopic policy
#     # and add its cost to the one-step lookahead cost

#     def compute_rollout(x: Array, y: Array, seed: Optional[int]) -> float:
#         return _rollout(
#             x,
#             y,
#             mdl,
#             horizon,
#             discount,
#             c1,
#             c2,
#             base_algorithm,
#             xl,
#             xu,
#             seed,
#             **minimize_kwargs,
#         )

#     a_rollout = parallel(
#         delayed(compute_rollout)(x, y, s) for x, y, s in zip(x, y_hat, make_seeds(seed))
#     )
#     return np.add(a, a_rollout)
