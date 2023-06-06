"""
Implementation of the non-myopic acquisition function for RBF/IDW-based Global
Optimization, based on the myopic function in [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.termination.default import DefaultSingleObjectiveTermination

from globopt.core.regression import Array, RegressorType, partial_fit, predict
from globopt.myopic.acquisition import acquisition as myopic_acquisition


def _rollout(
    x: Array,
    y_hat: Array,
    mdl: RegressorType,
    horizon: int,
    discount: float,
    c1: float,
    c2: float,
    algorithm: Algorithm,
    xl: Optional[npt.ArrayLike],
    xu: Optional[npt.ArrayLike],
    **minimize_kwargs: Any,
) -> float:
    """Rollouts the base greedy/myopic policy from the given point, using the regression
    to predict the evolution of the dynamics of the optimization problem."""
    n_var = x.size
    mdl = partial_fit(mdl, x.reshape(1, n_var), y_hat.reshape(1))
    y_min = mdl.ym_.min()
    y_max = mdl.ym_.max()
    a = 0.0
    for h in range(1, horizon):
        dym = y_max - y_min
        problem = FunctionalProblem(
            n_var,
            lambda x_: myopic_acquisition(x_, mdl, None, dym, c1, c2),
            xl=xl,
            xu=xu,
            elementwise=False,
        )
        res = minimize(problem, algorithm, verbose=False, **minimize_kwargs).opt[0]
        a += res.F.item() * discount**h

        # add new point to the regression model
        x = res.X.reshape(1, n_var)
        y_hat = predict(mdl, x)
        mdl = partial_fit(mdl, x, y_hat)
        y_min = min(y_min, y_hat)
        y_max = max(y_max, y_hat)
    return a


def acquisition(
    x: Array,
    mdl: RegressorType,
    horizon: int,
    discount: float = 1.0,
    c1: float = 1.5078,
    c2: float = 1.4246,
    base_algorithm: Algorithm = None,
    xl: Optional[npt.ArrayLike] = None,
    xu: Optional[npt.ArrayLike] = None,
    parallel: Parallel = None,
    **minimize_kwargs: Any,
) -> Array:
    """Computes the non-myopic acquisition function for IDW/RBF regression models.

    Parameters
    ----------
    x : array of shape (n_samples, n_var)
        Array of points for which to compute the acquisition. `n_samples` is the number
        of target points for which to compute the acquisition, and `n_var` is the number
        of features/variables of each point.
    mdl : RegressorType
        Fitted model to use for computing the acquisition function.
    horizon : int
        Length of the lookahead/non-myopic horizon.
    c1 : float, optional
        Weight of the contribution of the variance function, by default `1.5078`.
    c2 : float, optional
        Weight of the contribution of the distance function, by default `1.4246`.
    discount : float, optional
        Discount factor for the lookahead horizon. By default, `1.0`.

    Returns
    -------
    array of shape (n_samples,)
        The non-myopic acquisition function computed for each `x`.
    """
    if parallel is None:
        parallel = Parallel(n_jobs=-1, verbose=0)  # 10 for debugging
    if base_algorithm is None:
        base_algorithm = PSO()
    if "termination" not in minimize_kwargs:
        minimize_kwargs["termination"] = DefaultSingleObjectiveTermination(
            ftol=1e-4, n_max_gen=300, period=10
        )

    # compute the cost associated to the one-step lookahead
    y_hat = predict(mdl, x)
    a = myopic_acquisition(x, mdl, y_hat, None, c1, c2)
    if horizon == 1:
        return a

    # for each sample, compute the rollout policy by rolling out the base myopic policy
    # and add its cost to the one-step lookahead cost
    serial_fun = lambda x, y: _rollout(
        x, y, mdl, horizon, discount, c1, c2, base_algorithm, xl, xu, **minimize_kwargs
    )
    a += np.asarray(parallel(delayed(serial_fun)(x_, y_) for x_, y_ in zip(x, y_hat)))
    return a
