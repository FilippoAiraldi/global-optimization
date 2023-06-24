"""
Implementation of the non-myopic acquisition function for RBF/IDW-based Global
Optimization, based on the myopic function in [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from itertools import product
from typing import Any, Optional, Union

import numba as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.stats.qmc import MultivariateNormalQMC
from vpso import vpso
from vpso.typing import Array1d, Array2d, Array3d

from globopt.core.regression import (
    RegressorType,
    nb_Idw,
    nb_Rbf,
    partial_fit,
    predict,
    repeat,
    repeat_along_first_axis,
)
from globopt.myopic.acquisition import _compute_acquisition as myopic_acquisition
from globopt.myopic.acquisition import _idw_variance, _idw_weighting

"""Seed that is used for COMMON random numbers generation."""
FIXED_SEED = 1909

"""Number of elements per batch to use for parallelization."""
BATCH_SIZE = 2**5


@nb.njit(
    [
        nb.types.Tuple((nb.int64, types[0], types[1][1], types[1][1]))(
            nb.float64[:, :, :],
            types[0],
            nb.int64,
            nb.bool_,
            types[1][0],
            types[1][0],
            nb.bool_,
        )
        for types in product(
            (nb_Rbf, nb_Idw),
            ((nb.float64[:], nb.float64[:, :]), (nb.types.none, nb.types.none)),
        )
    ],
    cache=True,
    nogil=True,
)
def _initialize(
    x: Array3d,
    mdl: RegressorType,
    horizon: int,
    rollout: bool,
    lb: Optional[Array1d],
    ub: Optional[Array1d],
    check: bool,
) -> tuple[int, RegressorType, Optional[Array2d], Optional[Array2d]]:
    """Initializes the quantities need for computing the acquisition function."""
    n_samples, sample_h, _ = x.shape
    if check:
        assert mdl.Xm_.shape[0] == 1, "regression model must be non-batched"
        if rollout:
            assert (
                ub is not None and lb is not None
            ), "upper and lower bounds must be provided for rollout"
            assert sample_h == 1, "x must have only one time step for rollout"
        else:
            assert (
                sample_h == horizon
            ), "x must have the same number of time steps as the horizon"

    mdl = repeat(mdl, n_samples)
    if lb is not None and ub is not None:
        lb_ = repeat_along_first_axis(np.expand_dims(lb, 0), n_samples)
        ub_ = repeat_along_first_axis(np.expand_dims(ub, 0), n_samples)
    else:
        lb_ = None
        ub_ = None
    return n_samples, mdl, lb_, ub_


def _next_query_point(
    x: Array3d,
    mdl: RegressorType,
    h: int,
    c1: float,
    c2: float,
    y_min: Array3d,
    y_max: Array3d,
    lb: Optional[Array1d],
    ub: Optional[Array1d],
    rollout: bool,
    pso_kwargs: dict[str, Any],
    seed: Union[None, int, np.random.Generator],
) -> Array3d:
    """Computes the next point to query. If the strategy is `"mpc"`, then the next point
    is just the next point in the trajectory. If the strategy is `"rollout"`, then, for
    `h=0`, the next point is the input, and for `h>0`, the next point is the
    minimizer of the myopic acquisition function, i.e., base policy."""
    if not rollout or h == 0:
        return x[:, h, np.newaxis, :]
    dym = y_max - y_min
    func = lambda x_: myopic_acquisition(
        x_, mdl.Xm_, mdl.ym_, c1, c2, mdl.exp_weighting, predict(mdl, x_), dym
    )[:, :, 0]
    return vpso(func, lb, ub, **pso_kwargs, seed=seed)[0][:, np.newaxis, :]


@nb.njit(
    [
        nb.types.Tuple(
            (types[0], nb.float64[:], nb.float64[:, :, :], nb.float64[:, :, :])
        )(
            nb.float64[:, :, :],
            types[0],
            nb.float64,
            nb.float64,
            nb.float64[:, :, :],
            nb.float64[:, :, :],
            types[1],
        )
        for types in product((nb_Rbf, nb_Idw), (nb.float64[:], nb.types.none))
    ],
    cache=True,
    nogil=True,
)
def _advance(
    x_next: Array3d,
    mdl: RegressorType,
    c1: float,
    c2: float,
    y_min: Array3d,
    y_max: Array3d,
    rng: Optional[Array1d],
) -> tuple[RegressorType, Array1d, Array3d, Array3d]:
    """Advances the regression model dynamically, and returns the stage cost."""
    # predict dynamics of the regression
    y_hat = predict(mdl, x_next)
    if rng is not None:
        std = _idw_variance(
            y_hat, mdl.ym_, _idw_weighting(x_next, mdl.Xm_, mdl.exp_weighting)
        )
        y_hat[:, 0, 0] += std[:, 0, 0] * rng

    # compute cost/reward, fit regression to the new point, and update min/max
    cost = myopic_acquisition(
        x_next, mdl.Xm_, mdl.ym_, c1, c2, mdl.exp_weighting, y_hat, y_max - y_min
    )[:, 0, 0]
    mdl_new = partial_fit(mdl, x_next, y_hat)
    y_min_new = np.minimum(y_min, y_hat)
    y_max_new = np.maximum(y_max, y_hat)
    return mdl_new, cost, y_min_new, y_max_new


def _compute_acquisition(
    x: Array3d,
    mdl: RegressorType,
    horizon: int,
    discount: float,
    c1: float,
    c2: float,
    lb: Optional[Array1d],
    ub: Optional[Array1d],
    n_samples: int,
    rollout: bool,
    pso_kwargs: dict[str, Any],
    seed: Union[None, int, np.random.Generator],
    rng: Optional[Array2d] = None,
) -> Array1d:
    """Actual computation of the non-myopic acquisition acquisition function."""
    a = np.zeros(n_samples, dtype=np.float64)
    y_min = mdl.ym_.min(1, keepdims=True)
    y_max = mdl.ym_.max(1, keepdims=True)
    for h in range(horizon):
        x_next = _next_query_point(
            x, mdl, h, c1, c2, y_min, y_max, lb, ub, rollout, pso_kwargs, seed
        )
        mdl, cost, y_min, y_max = _advance(
            x_next, mdl, c1, c2, y_min, y_max, rng[h] if rng is not None else None
        )
        a += (discount**h) * cost
    return a


def deterministic_acquisition(
    x: Array3d,
    mdl: RegressorType,
    horizon: int,
    discount: float,
    c1: float = 1.5078,
    c2: float = 1.4246,
    rollout: bool = True,
    lb: Optional[Array1d] = None,
    ub: Optional[Array1d] = None,
    pso_kwargs: Optional[dict[str, Any]] = None,
    check: bool = True,
    seed: Union[None, int, np.random.Generator] = None,
) -> Array1d:
    """Computes the non-myopic acquisition function for IDW/RBF regression models with
    deterministic evolution of the regressor, i.e., without MC integration.

    Parameters
    ----------
    x : array of shape (n_samples, horizon, dim) or (n_samples, 1, dim)
        Array of points for which to compute the acquisition. `n_samples` is the number
        of target points for which to compute the acquisition, and `dim` is the number
        of features/variables of each point. In case of `rollout = False`, `horizon` is
        the length of the prediced trajectory of sampled points; while in case of
        `rollout == True"`, this dimension has to be 1, since only the first sample
        point is optimized over.
    mdl : Idw or Rbf
        Fitted model to use for computing the acquisition function.
    horizon : int
        Length of the prediced trajectory of sampled points.
    discount : float
        Discount factor for the lookahead horizon.
    c1 : float, optional
        Weight of the contribution of the variance function, by default `1.5078`.
    c2 : float, optional
        Weight of the contribution of the distance function, by default `1.4246`.
    rollout : bool, optional
        The strategy to be used for approximately solving the optimal control problem.
        If `True`, rollout is used where only the first sample point is optimized and
        then the remaining samples in the horizon are computed via the myopic
        acquisition base policy. If `False`, the whole horizon trajectory is optimized
        in an MPC fashion.
    lb, ub : 1d array, optional
        Lower and upper bounds of the search domain. Only required when
        `rollout == True`.
    ps_kwargs : dict, optional
        Options to be passed to the `vpso` solver. Cannot include `"seed"` key.
    check : bool, optional
        Whether to perform checks on the inputs, by default `True`.
    seed : int or np.random.Generator, optional
        Seed for the random number generator or a generator itself, by default `None`.

    Returns
    -------
    1d array
        The deterministic non-myopic acquisition function for each target point.
    """
    if pso_kwargs is None:
        pso_kwargs = {}
    n_samples, mdl, lb, ub = _initialize(x, mdl, horizon, rollout, lb, ub, check)
    seed = np.random.default_rng(seed)
    return _compute_acquisition(
        x, mdl, horizon, discount, c1, c2, lb, ub, n_samples, rollout, pso_kwargs, seed
    )


def _draw_standard_normal_sample(
    mc_iters: int,
    horizon: int,
    n_samples: int,
    quasi_mc: bool,
    common_random_numbers: bool,
    antithetic_variates: bool,
    seed: Union[None, int, np.random.Generator],
) -> tuple[Array3d, np.random.Generator]:
    """Draws a (quasi) random sample from the standard normal distribution, optionally
    with Quasi-MC, CRN, and antithetic variates."""
    if common_random_numbers:
        seed = FIXED_SEED  # overwrite seed
        n_samples = 1  # draw same numbers for all samples
    np_random = np.random.default_rng(seed)

    n = mc_iters // 2 if antithetic_variates else mc_iters
    shape = (n, horizon, n_samples)

    if quasi_mc:
        qmc_sampler = MultivariateNormalQMC(
            mean=np.zeros(horizon * n_samples), inv_transform=False, seed=np_random
        )
        sample = qmc_sampler.random(2 ** int(np.ceil(np.log2(n))))[:n].reshape(shape)
    else:
        sample = np_random.standard_normal(shape)

    if antithetic_variates:
        sample = np.concatenate((sample, -sample), 0)
    return sample, np_random


def acquisition(
    x: Array3d,
    mdl: RegressorType,
    horizon: int,
    discount: float,
    c1: float = 1.5078,
    c2: float = 1.4246,
    rollout: bool = True,
    lb: Optional[Array1d] = None,
    ub: Optional[Array1d] = None,
    #
    mc_iters: int = 1024,
    quasi_mc: bool = True,
    common_random_numbers: bool = True,
    antithetic_variates: bool = True,
    # control_variate: bool = True,  # TODO:
    #
    pso_kwargs: Optional[dict[str, Any]] = None,
    check: bool = True,
    seed: Union[None, int, np.random.Generator] = None,
    parallel: Union[None, Parallel, dict[str, Any]] = None,
    return_as_list: bool = False,
) -> Union[Array1d, list[Array1d]]:
    """Computes the non-myopic acquisition function for IDW/RBF regression models with
    Monte Carlo simulations.

    Parameters
    ----------
    x : array of shape (n_samples, horizon, dim) or (n_samples, 1, dim)
        Array of points for which to compute the acquisition. `n_samples` is the number
        of target points for which to compute the acquisition, and `dim` is the number
        of features/variables of each point. In case of `rollout = False`, `horizon` is
        the length of the prediced trajectory of sampled points; while in case of
        `rollout == True"`, this dimension has to be 1, since only the first sample
        point is optimized over.
    mdl : Idw or Rbf
        Fitted model to use for computing the acquisition function.
    horizon : int
        Length of the prediced trajectory of sampled points.
    discount : float
        Discount factor for the lookahead horizon.
    c1 : float, optional
        Weight of the contribution of the variance function, by default `1.5078`.
    c2 : float, optional
        Weight of the contribution of the distance function, by default `1.4246`.
    rollout : bool, optional
        The strategy to be used for approximately solving the optimal control problem.
        If `True`, rollout is used where only the first sample point is optimized and
        then the remaining samples in the horizon are computed via the myopic
        acquisition base policy. If `False`, the whole horizon trajectory is optimized
        in an MPC fashion.
    lb, ub : 1d array, optional
        Lower and upper bounds of the search domain. Only required when
        `rollout == True`.
    mc_iters : int, optional
        Number of Monte Carlo iterations, by default `1024`. For better sampling, the
        iterations should be a power of 2.
    quasi_mc : bool, optional
        Whether to use quasi Monte Carlo sampling, by default `True`.
    common_random_numbers : bool, optional
        Whether to use common random numbers, by default `True`. In this case, `seed`,
        if passed at all, is discarded.
    antithetic_variates : bool, optional
        Whether to use antithetic variates, by default `True`.
    control_variate  # TODO
    pso_kwargs : dict, optional
        Options to be passed to the `vpso` solver. Cannot include `"seed"` key.
    check : bool, optional
        Whether to perform checks on the inputs, by default `True`.
    seed : int or np.random.Generator, optional
        Seed for the random number generator or a generator itself, by default `None`.
        Only used when `common_random_numbers == False`.
    parallel : Parallel or dict, optional
        Parallelization of MC iterations. If an instance of `Parallel` is passed, it is
        used to parallelize the loop. If a dictionary is passed, it is used as kwargs to
        instantiate a `Parallel` object. If `None`, no parallelization is performed.
    return_as_list : bool, optional
        Whether to return the list of iterations of the MC integration or just the
        resulting average. By default `False`, which only returns the average.

    Returns
    -------
    1d array or list of
        The non-myopic acquisition function for each target point (per MC iteration,
        if `return_as_list == True`)
    """
    if pso_kwargs is None:
        pso_kwargs = {}
    n_samples, mdl, lb, ub = _initialize(x, mdl, horizon, rollout, lb, ub, check)

    # create random generator and draw all the necessary random numbers
    random_numbers, np_random = _draw_standard_normal_sample(
        mc_iters,
        horizon,
        n_samples,
        quasi_mc,
        common_random_numbers,
        antithetic_variates,
        seed,
    )
    seeds = np_random.bit_generator._seed_seq.spawn(mc_iters)  # type: ignore

    # instantiate the parallel object
    if parallel is None:
        parallel = Parallel(n_jobs=1, verbose=0)
    elif isinstance(parallel, dict):
        parallel = Parallel(**parallel)

    # run the MC iterations, possibly in paralle
    results = parallel(
        delayed(_compute_acquisition)(
            x,
            mdl,
            horizon,
            discount,
            c1,
            c2,
            lb,
            ub,
            n_samples,
            rollout,
            pso_kwargs,
            seeds[i],
            random_numbers[i],
        )
        for i in range(mc_iters)
    )
    return (
        results
        if return_as_list
        else sum(results, start=np.zeros(n_samples)) / mc_iters
    )


# TODO:
# 1. implement control variate
