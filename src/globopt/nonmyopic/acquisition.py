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
from globopt.myopic.acquisition import acquisition as myopic_acquisition
from globopt.myopic.acquisition import _idw_variance, _idw_weighting

"""Seed that is used for COMMON random numbers generation."""
FIXED_SEED = 1909


@nb.njit(
    [
        nb.types.Tuple(
            (
                nb.float64[:],
                mdl_type,
                nb.float64[:, :],
                nb.float64[:, :],
                nb.float64[:, :, :],
                nb.float64[:, :, :],
            )
        )(
            nb.float64[:, :, :],
            mdl_type,
            nb.int64,
            nb.float64,
            nb.float64,
            nb.float64[:],
            nb.float64[:],
        )
        for mdl_type in (nb_Rbf, nb_Idw)
    ],
    cache=True,
    nogil=True,
)
def _compute_myopic_cost(
    x_trajectory: Array3d,
    mdl: RegressorType,
    n_samples: int,
    c1: float,
    c2: float,
    lb: Array1d,
    ub: Array1d,
) -> tuple[Array1d, RegressorType, Array3d, Array3d, Array3d, Array3d]:
    """Computes the first step of the non-myopic acquisition function (which uses the
    starting common regressor and is not dependent on the number of MC iterations) and
    returns the associated cost and batch-dimension-expanded models."""
    y_min = mdl.ym_.min()  # ym_ ∈ (1, n_samples, 1)
    y_max = mdl.ym_.max()
    dym = np.full((1, 1, 1), y_max - y_min)
    x_next = x_trajectory[np.newaxis, :, 0, :]  # take first element in the horizon
    cost = myopic_acquisition(x_next, mdl, c1, c2, None, dym)[0, :, 0]  # ∈ (n_samples,)

    mdl_ = repeat(mdl, n_samples)
    lb_ = repeat_along_first_axis(np.expand_dims(lb, 0), n_samples)
    ub_ = repeat_along_first_axis(np.expand_dims(ub, 0), n_samples)
    y_min_ = np.full((n_samples, 1, 1), y_min)
    y_max_ = np.full((n_samples, 1, 1), y_max)
    return cost, mdl_, lb_, ub_, y_min_, y_max_


@nb.njit(
    [
        nb.types.Tuple(
            (mdl_type, nb.float64[:, :, :], nb.float64[:, :, :], nb.float64[:, :, :])
        )(
            nb.float64[:, :, :],
            mdl_type,
            nb.float64[:, :, :],
            nb.float64[:, :, :],
            rng_type,
        )
        for mdl_type, rng_type in product(
            (nb_Rbf, nb_Idw), (nb.float64[:], nb.types.none)
        )
    ],
    cache=True,
    nogil=True,
)
def _advance(
    x_next: Array3d,
    mdl: RegressorType,
    y_min: Array3d,
    y_max: Array3d,
    prediction_rng: Optional[Array1d],
) -> tuple[RegressorType, Array3d, Array3d, Array3d]:
    """Predicts the function value at the next point and (deterministically or
    stochastically) advances the regression model's dynamics and the y-delta."""
    y_hat = predict(mdl, x_next)
    if prediction_rng is not None:
        W = _idw_weighting(x_next, mdl.Xm_, mdl.exp_weighting)
        std = _idw_variance(y_hat, mdl.ym_, W)
        y_hat[:, 0, 0] += std[:, 0, 0] * prediction_rng

    mdl_ = partial_fit(mdl, x_next, y_hat)
    y_min_ = np.minimum(y_min, y_hat)
    y_max_ = np.maximum(y_max, y_hat)
    return mdl_, y_min_, y_max_, y_max_ - y_min_


def _next_query_point(
    x: Array3d,
    mdl: RegressorType,
    current_step: int,  # > 0
    c1: float,
    c2: float,
    dym: Array3d,
    lb: Array2d,
    ub: Array2d,
    rollout: bool,
    pso_kwargs: dict[str, Any],
    np_random: np.random.Generator,
) -> tuple[Array3d, Array1d]:
    """Computes the next point to query and its associated cost. If the strategy is
    `"mpc"`, then the next point is just the next point in the trajectory. If the
    strategy is `"rollout"`, then the next point is the minimizer of the myopic
    acquisition function, i.e., base policy."""
    if not rollout:
        x_next = x[:, current_step, np.newaxis, :]  # ∈ (n_samples, 1, dim)
        return x_next, myopic_acquisition(x_next, mdl, c1, c2, None, dym)[:, 0, 0]

    def func(q: Array3d) -> Array2d:
        return myopic_acquisition(q, mdl, c1, c2, None, dym)[:, :, 0]

    x_next, cost, _ = vpso(func, lb, ub, **pso_kwargs, seed=np_random)
    return x_next[:, np.newaxis, :], cost


def _terminal_cost(
    mdl: RegressorType,
    lb: Array1d,
    ub: Array1d,
    pso_kwargs: dict[str, Any],
    np_random: np.random.Generator,
) -> Array1d:
    """Computes the terminal cost of the sample trajectory as the greedy minimum of the
    predicted function, with no exploration."""
    return vpso(lambda x: predict(mdl, x), lb, ub, **pso_kwargs, seed=np_random)[1]


def _compute_nonmyopic_cost(
    x_trajectory: Array3d,
    mdl: RegressorType,
    n_samples: int,
    horizon: int,
    discount: float,
    y_min: Array3d,
    y_max: Array3d,
    c1: float,
    c2: float,
    lb: Array1d,
    ub: Array1d,
    rollout: bool,
    pso_kwargs: dict[str, Any],
    prediction_rng: Optional[Array2d],
    seed: Union[None, int, np.random.Generator],
) -> Array1d:
    """After the first myopic tep, computes the cost along the remaining horizon by
    predicting the function value at the next point and updating the regression model's
    dynamics with such predictions."""
    np_random = np.random.default_rng(seed)
    cost = np.zeros(n_samples)
    x_next = x_trajectory[:, 0, np.newaxis, :]  # ∈ (n_samples, 1, n_features)
    h = 0
    while True:
        rng = prediction_rng[h] if prediction_rng is not None else None
        mdl, y_min, y_max, dym = _advance(x_next, mdl, y_min, y_max, rng)
        h += 1
        if h >= horizon:
            break
        x_next, current_cost = _next_query_point(
            x_trajectory, mdl, h, c1, c2, dym, lb, ub, rollout, pso_kwargs, np_random
        )
        cost += (discount**h) * current_cost  # accumulate in-place
    cost += _terminal_cost(mdl, lb, ub, pso_kwargs, np_random)
    return cost


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
    lb: Array1d,
    ub: Array1d,
    c1: float = 1.5078,
    c2: float = 1.4246,
    rollout: bool = True,
    #
    mc_iters: int = 1024,
    quasi_mc: bool = True,
    common_random_numbers: bool = True,
    antithetic_variates: bool = True,
    #
    pso_kwargs: Optional[dict[str, Any]] = None,
    check: bool = True,
    seed: Union[None, int, np.random.Generator] = None,
    parallel: Union[None, Parallel, dict[str, Any]] = None,
    return_iters: bool = False,
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
        Length of the prediced trajectory of sampled points. Note that if `horizon=0`,
        this acquisition function does not fall back to the myopic version, since it
        takes into account the final terminal cost.
    lb, ub : 1d array of shape (dim,)
        Lower and upper bounds of the search domain.
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
    mc_iters : int, optional
        Number of Monte Carlo iterations, by default `1024`. For better sampling, the
        iterations should be a power of 2. If `0`, the acquisition is computed
        deterministically.
    quasi_mc : bool, optional
        Whether to use quasi Monte Carlo sampling, by default `True`.
    common_random_numbers : bool, optional
        Whether to use common random numbers, by default `True`. In this case, `seed`,
        if passed at all, is discarded.
    antithetic_variates : bool, optional
        Whether to use antithetic variates, by default `True`.
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
    return_iters : bool, optional
        Whether to return the list of iterations of the MC integration or just the
        resulting average. By default `False`, which only returns the average.

    Returns
    -------
    1d array or list of
        The non-myopic acquisition function for each target point (per MC iteration,
        if `return_as_list == True`)
    """
    n_samples, h_, _ = x.shape
    if check:
        assert mdl.Xm_.shape[0] == 1, "regression model must be non-batched"
        if rollout:
            assert h_ == 1, "x must have only one time step for rollout"
        else:
            assert (
                h_ == horizon
            ), "x must have the same number of time steps as the horizon"
    if pso_kwargs is None:
        pso_kwargs = {}

    # compute the first (myopic) iteration of the acquisition function - this is in
    # common across all MC iterations
    myopic_cost, mdl, lb, ub, y_min, y_max = _compute_myopic_cost(
        x, mdl, n_samples, c1, c2, lb, ub
    )

    # if no mc_iters, solve the problem deterministically
    if mc_iters <= 0:
        np_random = np.random.default_rng(seed)
        nonmyopic_cost = _compute_nonmyopic_cost(
            x,
            mdl,
            n_samples,
            horizon,
            discount,
            y_min,
            y_max,
            c1,
            c2,
            lb,
            ub,
            rollout,
            pso_kwargs,
            None,
            np_random,
        )
        return myopic_cost + nonmyopic_cost

    # draw the standard normal samples for the prediction noise. These can be random
    # or pseudo-random (QMC), and can be antithetic or not.
    prediction_random_numbers, np_random = _draw_standard_normal_sample(
        mc_iters,
        horizon,
        n_samples,
        quasi_mc,
        common_random_numbers,
        antithetic_variates,
        seed,
    )
    seeds = np_random.bit_generator._seed_seq.spawn(mc_iters)  # type: ignore

    # run the MC iterations, possibly in parallel
    if parallel is None:
        parallel = Parallel(n_jobs=1, verbose=0)
    elif isinstance(parallel, dict):
        parallel = Parallel(**parallel)
    nonmyopic_costs = parallel(
        delayed(_compute_nonmyopic_cost)(
            x,
            mdl,
            n_samples,
            horizon,
            discount,
            y_min,
            y_max,
            c1,
            c2,
            lb,
            ub,
            rollout,
            pso_kwargs,
            prediction_random_numbers[i],
            seeds[i],
        )
        for i in range(mc_iters)
    )
    if return_iters:
        return [myopic_cost + nonmyopic_cost for nonmyopic_cost in nonmyopic_costs]
    return myopic_cost + sum(nonmyopic_costs) / mc_iters
