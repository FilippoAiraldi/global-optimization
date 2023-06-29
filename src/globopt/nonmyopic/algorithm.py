"""
Implementation of non-myopic version of the Global Optimization strategy based on RBF or
IDW regression from [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from joblib import Parallel
from numpy.typing import ArrayLike
from vpso import vpso
from vpso.typing import Array1d, Array2d

from globopt.core.regression import Idw, Rbf
from globopt.myopic.algorithm import _advance, _initialize
from globopt.nonmyopic.acquisition import acquisition


def _next_query_point(
    mdl: Union[Idw, Rbf],
    lb: Array1d,
    ub: Array1d,
    dim: int,
    horizon: int,
    discount: float,
    c1: float,
    c2: float,
    rollout: bool,
    mc_iters: int,
    quasi_mc: bool,
    common_random_numbers: bool,
    antithetic_variates: bool,
    parallel: Parallel,
    iteration: int,
    np_random: np.random.Generator,
    pso_kwargs: Optional[dict[str, Any]],
) -> tuple[Array1d, float]:
    """Computes the next point to query by minimizing the acquisition function."""
    if rollout:
        h_ = 1
        lb_acquisition = lb[0]
        ub_acquisition = ub[0]
    else:
        h_ = horizon
        lb = lb[:, : h_ * dim]  # due to shrinking horizon
        ub = ub[:, : h_ * dim]
        lb_acquisition = lb[0, :dim]
        ub_acquisition = ub[0, :dim]
    check_acquisition = iteration == 1
    vpso_func = lambda x: acquisition(
        x.reshape(-1, h_, dim),
        mdl,
        horizon,
        discount,
        lb_acquisition,
        ub_acquisition,
        c1,
        c2,
        rollout,
        mc_iters,
        quasi_mc,
        common_random_numbers,
        antithetic_variates,
        pso_kwargs,
        check_acquisition,
        np_random,
        parallel,
        False,
    )
    x_new, acq_opt, _ = vpso(vpso_func, lb, ub, seed=np_random, **pso_kwargs)
    return x_new[0, :dim], acq_opt.item()


def nmgo(
    func: Callable[[Array2d], ArrayLike],
    lb: Array1d,
    ub: Array1d,
    mdl: Union[Idw, Rbf],
    horizon: int,
    discount: float,
    init_points: Union[int, Array2d] = 5,
    c1: float = 1.5078,
    c2: float = 1.4246,
    #
    rollout: bool = True,
    mc_iters: int = 1024,
    quasi_mc: bool = True,
    common_random_numbers: bool = True,
    antithetic_variates: bool = True,
    parallel: Union[None, Parallel, dict[str, Any]] = None,
    #
    maxiter: int = 50,
    seed: Union[None, int, np.random.Generator] = None,
    callback: Optional[Callable[[Literal["go", "nmgo"], dict[str, Any]], None]] = None,
    pso_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Array1d, float]:
    """Non-Myopic Global Optimization (NM-GO) based on RBF or IDW regression [1].

    Parameters
    ----------
    func : callable with 2d array adn returns array_like
        The (possibly, unknown) function to be minimized via GO. It must take as input
        a 2d array of shape `(n_points, dim)` and return a 1d array_like of shape. The
        input represents the points at which the function must be evaluated, while the
        output is the value of the function at those points.
    lb, ub : 1d array
        Lower and upper bounds of the search space. The dimensionality of the search
        space is inferred from the size of `lb` and `ub`.
    mdl : Idw or Rbf
        The regression model to be used in the approximation of the unknown function.
    horizon : int
        Length of the prediced trajectory of sampled points. Note that if `horizon=0`,
        this acquisition function does not fall back to the myopic version, since it
        takes into account the final terminal cost.
    discount : float
        Discount factor for the lookahead horizon.
    init_points : int or 2d array, optional
        The initial points of the problem. If an integer is passed, the points are
        sampled via latin hypercube sampling over the search space. If a 2d array is
        passed, then these are used as initial points. By default, `5`.
    c1 : float, optional
        Weight of the contribution of the variance function in the algorithm's
        acquisition function, by default `1.5078`.
    c2 : float, optional
        Weight of the contribution of the distance function in the algorithm's
        acquisition function, by default `1.4246`.
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
    parallel : Parallel or dict, optional
        Parallelization of MC iterations. If an instance of `Parallel` is passed, it is
        used to parallelize the loop. If a dictionary is passed, it is used as kwargs to
        instantiate a `Parallel` object. If `None`, no parallelization is performed.
    maxiter : int, optional
        Maximum number of iterations to run the algorithm for, by default `50`.
    seed : int or generator, optional
        Seed for the random number generator or a generator itself, by default `None`.
    callback : callable with {"go", "nmgo"} and dict, optional
        A callback function called before the first and at the end of each iteration.
        It must take as input a string indicating the current algorithm and a dictionary
        with the local variables of the algorithm. By default, `None`.
    pso_kwargs : dict, optional
        Optional keyword arguments to be passed to the `vpso` algorithm used to optimize
        the acquisition function.

    Returns
    -------
    tuple of 1d array and float
        Returns a tuple containing the best minimizer and minimum found by the end of
        the iterations.

    References
    ----------
    [1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
        functions. Computational Optimization and Applications, 77(2):571–595, 2020
    """
    mdl, lb, ub, dim, x_best, y_best, y_min, y_max, pso_kwargs, np_random = _initialize(
        mdl, func, lb, ub, init_points, seed, pso_kwargs
    )
    if callback is not None:
        callback("nmgo", locals())

    if not rollout:
        lb = np.tile(lb, (1, horizon))
        ub = np.tile(ub, (1, horizon))
    if parallel is None:
        parallel = Parallel(n_jobs=1, verbose=0)
    elif isinstance(parallel, dict):
        parallel = Parallel(**parallel)

    for iteration in range(1, maxiter + 1):
        x_new, acq_opt = _next_query_point(
            mdl,
            lb,
            ub,
            dim,
            horizon,
            discount,
            c1,
            c2,
            rollout,
            mc_iters,
            quasi_mc,
            common_random_numbers,
            antithetic_variates,
            parallel,
            iteration,
            np_random,
            pso_kwargs,
        )
        y_new = float(func(x_new[np.newaxis]))
        mdl_new, x_best, y_best, y_min, y_max = _advance(
            mdl, x_new, y_new, x_best, y_best, y_min, y_max
        )
        if callback is not None:
            callback("nmgo", locals())
        mdl = mdl_new
        horizon = min(horizon, maxiter - iteration)

    return x_best, y_best
