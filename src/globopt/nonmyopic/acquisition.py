"""
Implementation of the non-myopic acquisition function for RBF/IDW-based Global
Optimization, based on the myopic function in [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from functools import partial
from itertools import islice, product
from typing import Any, Iterable, Iterator

import numpy as np
from joblib import Parallel, delayed
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from globopt.core.regression import Array, RegressorType, partial_fit, predict, repeat
from globopt.myopic.acquisition import acquisition as myopic_acquisition


def acquisition(
    x: Array,
    mdl: RegressorType,
    c1: float = 1.5078,
    c2: float = 1.4246,
    discount: float = 1.0,
) -> Array:
    """Computes the non-myopic acquisition function for IDW/RBF regression models.

    Parameters
    ----------
    x : array of shape (n_samples, horizon, n_var)
        Array of trajectories for which to compute the acquisition function. `n_samples`
        is the number of units processed in parallel, `horizon` is the horizon length of
        the trajectories, and `n_var` is the number of features/variables per point in
        each trajectory.
    mdl : RegressorType
        Fitted model to use for computing the acquisition function.
    c1 : float, optional
        Weight of the contribution of the variance function, by default `1.5078`.
    c2 : float, optional
        Weight of the contribution of the distance function, by default `1.4246`.
    discount : float, optional
        Discount factor for the lookahead horizon. By default, `1.0`.

    Returns
    -------
    array of shape (n_samples,)
        The non-myopic acquisition function computed for each trajectory in input `x`.
    """
    n_samples, horizon, _ = x.shape

    # repeat the regressor along the sample dim
    mdl = repeat(mdl, n_samples)

    # loop over the horizon
    a = np.zeros(n_samples, dtype=float)
    for h in range(horizon):
        x_h = x[:, h, np.newaxis, :]
        y_hat = predict(mdl, x_h)
        a_h = myopic_acquisition(x_h, mdl, y_hat, None, c1, c2)
        a += (discount**h) * a_h[:, 0, 0]
        mdl = partial_fit(mdl, x_h, y_hat)
    return a


def optimal_acquisition(
    x: Array,
    mdl: RegressorType,
    h: int,
    c1: float = 1.5078,
    c2: float = 1.4246,
    discount: float = 1.0,
    brute_force: bool = True,
    verbosity: int = 0,
) -> Array:
    """Helper function to compute the optimal non-myopic acquisition function.

    Parameters
    ----------
    x : array of shape (n_samples, n_var)
        Array of decision variables `x` for which to compute the acquisition.
        `n_samples` and `n_var` is the number of features/variables per point in `x`.
    mdl : RegressorType
        The fitted model to use for computing the acquisition function.
    h : int
        Horizon over which to project `x` and compute the acquisition function.
    c1 : float, optional
        Weight of the contribution of the variance function, by default `1.5078`.
    c2 : float, optional
        Weight of the contribution of the distance function, by default `1.4246`.
    discount : float, optional
        Discount factor for the lookahead horizon. By default, `1.0`.
    brute_force : bool, optional
        If `True`, the acquisition function will be evaluated for the `n_samples**h`
        trajectories obtained via cartesian product, and the optimal acquisition for the
        first decision variable will be returned. If `False`, instead of the brute force
        search, the acquisition function is computed by fixing each `x` as the first
        decision variable and computing the optimal remaining trajectory with `PSO`. In
        both cases, processing is parallelized in an effort to speed up the computation.
        By default, `False`.
    verbosity : int, optional
        Verbosity level for the parallel processing, by default `0`.

    Returns
    -------
    array of shape (n_samples,)
        The optimal non-myopic acquisition function computed for each trajectory
        starting from each point in `x`.
    """
    if h == 1:
        return myopic_acquisition(np.expand_dims(x, 0), mdl, c1=c1, c2=c2).reshape(-1)
    return (  # type: ignore[operator]
        _optimal_acquisition_by_brute_force
        if brute_force
        else _optimal_acquisition_by_minimization
    )(x, mdl, h, c1, c2, discount, verbosity)


def _batched(iterable: Iterable[Any], n: int) -> Iterator[Any]:
    # taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def _optimal_acquisition_by_brute_force(
    x: Array,
    mdl: RegressorType,
    h: int,
    c1: float,
    c2: float,
    discount: float,
    verbosity: int,
    chunk_size: int = 2**11,
) -> Array:
    """Utility to compute the optimal acquisition function by brute force search."""
    # enumerate all trajectories of length h (n_trajectories, h, n_var), evaluate the
    # acquisition function for each trajectory, and return the best for each x. The
    # computations are sped up by chunkifying the trajectories and evaluating the
    # acquisition function for each chunk in parallel

    n_samples = x.shape[0]
    trajectories = product(*(x for _ in range(h)))
    chunks = map(
        np.asarray, _batched(trajectories, chunk_size)
    )  # n_traj = n_samples**h

    fun = partial(acquisition, mdl=mdl, c1=c1, c2=c2, discount=discount)
    a_chunks = Parallel(n_jobs=-1, verbose=verbosity)(delayed(fun)(c) for c in chunks)
    a = np.concatenate(a_chunks, 0)
    return a.reshape(n_samples, n_samples ** (h - 1)).min(1)


def _optimal_acquisition_by_minimization(
    x: Array,
    mdl: RegressorType,
    h: int,
    c1: float,
    c2: float,
    discount: float,
    verbosity: int,
) -> Array:
    """Utility to compute the optimal acquisition function by PSO minimization."""
    # in a for loop, fix each x as the first decision variable and compute the optimal
    # remaining trajectory with PSO
    n_samples, n_var = x.shape
    x_lb, x_ub = x.min(0), x.max(0)
    if n_var == 1:
        x_lb, x_ub = x_lb.item(), x_ub.item()  # without this, it crashes
    pop_size = 20 * (h - 1)

    def obj(x_first: Array, x_: Array) -> Array:
        # reshape from (pop_size, n_var * (h - 1)) to (pop_size, h - 1, n_var) and
        # append fixed firt decision variable
        x_ = np.concatenate((x_first, x_.reshape(pop_size, h - 1, n_var)), 1)
        return acquisition(x_, mdl, c1, c2, discount)

    def solve_for_one_x(x_first: Array) -> float:
        x_first = np.broadcast_to(x_first, (pop_size, 1, n_var))
        obj_ = partial(obj, x_first)
        problem = FunctionalProblem(
            n_var=n_var * (h - 1), objs=obj_, xl=x_lb, xu=x_ub, elementwise=False
        )
        return minimize(problem, PSO(pop_size)).opt[0].F.item()

    a = Parallel(n_jobs=-1, verbose=verbosity)(
        delayed(solve_for_one_x)(x[i]) for i in range(n_samples)
    )
    return np.asarray(a)
