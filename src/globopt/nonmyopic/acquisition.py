"""
Implementation of the non-myopic acquisition function for RBF/IDW-based Global
Optimization, based on the myopic function in [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from typing import Any, Literal, Optional, Union

import numba as nb
import numpy as np
from scipy.stats.qmc import MultivariateNormalQMC
from vpso import vpso
from vpso.typing import Array1d, Array2d, Array3d

from globopt.core.regression import (
    RegressorType,
    partial_fit,
    predict,
    repeat,
    repeat_along_first_axis,
)
from globopt.myopic.acquisition import _idw_variance, _idw_weighting
from globopt.myopic.acquisition import acquisition as myopic_acquisition

"""Seed that is used for COMMON random numbers generation."""
FIXED_SEED = 1909


@nb.njit(
    [
        nb.types.void(
            nb.float64[:, :, :],
            nb.int64,
            nb.int64,
            nb.types.unicode_type,
            nb.types.none,
            nb.types.none,
        ),
        nb.types.void(
            nb.float64[:, :, :],
            nb.int64,
            nb.int64,
            nb.types.unicode_type,
            nb.float64[:],
            nb.float64[:],
        ),
    ],
    cache=True,
    nogil=True,
)
def _check_args(
    x: Array3d,
    batch: int,
    horizon: int,
    type: Literal["rollout", "mpc"],
    lb: Optional[Array1d],
    ub: Optional[Array1d],
) -> None:
    """Checks input arguments."""
    assert batch == 1, "regression model must be non-batched"
    if type == "rollout":
        assert (
            ub is not None and lb is not None
        ), "upper and lower bounds must be provided for rollout"
        assert x.shape[1] == 1, "x must have only one time step for rollout"
    else:
        assert (
            x.shape[1] == horizon
        ), "x must have the same number of time steps as the horizon"


@nb.njit(cache=True, nogil=True)
def _initialize_mdl_and_bounds(
    x: Array3d,
    mdl: RegressorType,
    type: Literal["rollout", "mpc"],
    lb: Optional[Array1d],
    ub: Optional[Array1d],
) -> tuple[int, RegressorType, Optional[Array2d], Optional[Array2d], bool]:
    """Initializes the quantities need for computing the acquisition function."""
    n_samples = x.shape[0]
    mdl = repeat(mdl, n_samples)
    rollout = type != "mpc"
    if lb is not None and ub is not None:
        lb_ = repeat_along_first_axis(np.expand_dims(lb, 0), n_samples)
        ub_ = repeat_along_first_axis(np.expand_dims(ub, 0), n_samples)
    else:
        lb_ = ub_ = None
    return n_samples, mdl, lb_, ub_, rollout


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
    seed: np.random.Generator,
) -> Array3d:
    """Computes the next point to query. If the strategy is `"mpc"`, then the next point
    is just the next point in the trajectory. If the strategy is `"rollout"`, then, for
    `h=0`, the next point is the input, and for `h>0`, the next point is the
    minimizer of the myopic acquisition function, i.e., base policy."""
    if not rollout:  # type == "mpc"
        return x[:, h, np.newaxis, :]
    elif h == 0:  # type == "rollout" and first iteration
        return x
    dym = y_max - y_min
    func = lambda x: myopic_acquisition(x, mdl, c1, c2, None, dym)[:, :, 0]
    return vpso(func, lb, ub, **pso_kwargs, seed=seed)[0][:, np.newaxis, :]


@nb.njit(cache=True, nogil=True)
def _advance(
    x: Array3d,
    x_next: Array3d,
    mdl: RegressorType,
    h: int,
    gamma: float,
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
            y_hat, mdl.ym_, _idw_weighting(x, mdl.Xm_, mdl.exp_weighting)
        )
        y_hat[:, 0, 0] += std[:, 0, 0] * rng

    # compute reward, fit regression to new point, and update min/max
    dym = y_max - y_min
    r = (gamma**h) * myopic_acquisition(x_next, mdl, c1, c2, y_hat, dym)[:, 0, 0]
    mdl = partial_fit(mdl, x_next, y_hat)
    y_min = np.minimum(y_min, y_hat)
    y_max = np.maximum(y_max, y_hat)
    return mdl, r, y_min, y_max


def _compute_acquisition(
    x: Array3d,
    mdl: RegressorType,
    horizon: int,
    gamma: float,
    c1: float,
    c2: float,
    lb: Optional[Array1d],
    ub: Optional[Array1d],
    n_samples: int,
    rollout: bool,
    pso_kwargs: dict[str, Any],
    seed: np.random.Generator,
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
        rng_h = rng[h] if rng is not None else None
        mdl, r, y_min, y_max = _advance(
            a, x, x_next, mdl, h, gamma, c1, c2, y_min, y_max, rng_h
        )
        a += r
    return a


def deterministic_acquisition(
    x: Array3d,
    mdl: RegressorType,
    horizon: int,
    discount: float,
    c1: float = 1.5078,
    c2: float = 1.4246,
    type: Literal["rollout", "mpc"] = "rollout",
    lb: Optional[Array1d] = None,  # only when `type == "rollout"`
    ub: Optional[Array1d] = None,  # only when `type == "rollout"`
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
        of features/variables of each point. In case of `type == "mpc"`, `horizon` is
        the length of the prediced trajectory of sampled points; while in case of
        `type == "rollout"`, since only the first sample point is optimized over, this
        dimension has to be 1.
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
    type : {"rollout", "mpc"}, optional
        The strategy to be used for approximately solving the optimal control problem.
        `"rollout"` optimizes over only the first sample point and then applies the
        myopic acquisition as base policy for the remaining horizon. `"mpc"` optimizes
        over the entire trajectory instead.
    lb, ub : 1d array, optional
        Lower and upper bounds of the search domain. Only required when
        `type == "rollout"`.
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
    if check:
        _check_args(x, mdl.Xm_.shape[0], horizon, type, lb, ub)
    seed = np.random.default_rng(seed)
    n_samples, mdl, lb, ub, rollout = _initialize_mdl_and_bounds(x, mdl, type, lb, ub)
    return _compute_acquisition(
        x, mdl, horizon, discount, c1, c2, lb, ub, n_samples, rollout, pso_kwargs, seed
    )


def _draw_normal(
    mc_iters: int,
    horizon: int,
    n_samples: int,
    quasi_mc: bool,
    np_random: np.random.Generator,
) -> Array3d:
    """Draws a (quasi) random sample from the standard normal distribution."""
    if not quasi_mc:
        np_random.standard_normal((mc_iters, horizon, n_samples))

    qmc_sampler = MultivariateNormalQMC(
        mean=np.zeros(horizon * n_samples), inv_transform=False, seed=np_random
    )
    n_ = 2 ** int(np.ceil(np.log2(mc_iters)))
    return qmc_sampler.random(n_)[:mc_iters].reshape((mc_iters, horizon, n_samples))


def acquisition(
    x: Array3d,
    mdl: RegressorType,
    horizon: int,
    discount: float,
    c1: float = 1.5078,
    c2: float = 1.4246,
    type: Literal["rollout", "mpc"] = "rollout",
    lb: Optional[Array1d] = None,  # only when `type == "rollout"`
    ub: Optional[Array1d] = None,  # only when `type == "rollout"`
    #
    mc_iters: int = 1024,
    quasi_mc: bool = True,
    common_random_numbers: bool = True,
    antithetic_variates: bool = True,
    # control_variate: bool = True,  # TODO:
    #
    pso_kwargs: Optional[dict[str, Any]] = None,
    check: bool = True,
    seed: Optional[int] = None,
    return_as_list: bool = False,
) -> Union[Array1d, list[Array1d]]:
    """Computes the non-myopic acquisition function for IDW/RBF regression models with
    Monte Carlo simulations.

    Parameters
    ----------
    x : array of shape (n_samples, horizon, dim) or (n_samples, 1, dim)
        Array of points for which to compute the acquisition. `n_samples` is the number
        of target points for which to compute the acquisition, and `dim` is the number
        of features/variables of each point. In case of `type == "mpc"`, `horizon` is
        the length of the prediced trajectory of sampled points; while in case of
        `type == "rollout"`, since only the first sample point is optimized over, this
        dimension has to be 1.
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
    type : {"rollout", "mpc"}, optional
        The strategy to be used for approximately solving the optimal control problem.
        `"rollout"` optimizes over only the first sample point and then applies the
        myopic acquisition as base policy for the remaining horizon. `"mpc"` optimizes
        over the entire trajectory instead.
    lb, ub : 1d array, optional
        Lower and upper bounds of the search domain. Only required when
        `type == "rollout"`.
    mc_iters : int, optional
        Number of Monte Carlo iterations, by default `1024`.
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
    if check:
        _check_args(x, mdl.Xm_.shape[0], horizon, type, lb, ub)
    n_samples, mdl, lb, ub, rollout = _initialize_mdl_and_bounds(x, mdl, type, lb, ub)

    # create random generator and draw all the necessary random numbers
    np_random = np.random.default_rng(
        FIXED_SEED if common_random_numbers else seed  # overwrite seed by CRN
    )
    if antithetic_variates:
        rng = _draw_normal(mc_iters // 2, horizon, n_samples, quasi_mc, np_random)
        rng = np.concatenate((rng, -rng), 0)
    else:
        rng = _draw_normal(mc_iters, horizon, n_samples, quasi_mc, np_random)

    # loop over MC iterations and return the average
    generator = (
        _compute_acquisition(
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
            np_random,
            rng[i],
        )
        for i in range(mc_iters)
    )
    return (
        list(generator)
        if return_as_list
        else sum(generator, start=np.zeros(n_samples)) / mc_iters
    )


# TODO:
# 1. parallelize MC loop with joblib (try threads and processes)
# 2. implement control variate
# 3. try to jit as much as possible
