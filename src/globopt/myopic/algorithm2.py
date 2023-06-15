"""
Implementation of myopic Global Optimization strategy based on RBF or IDW regression.
The scheme was first proposed in [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


import logging
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats.qmc import LatinHypercube
from typing_extensions import TypeAlias
from vpso import vpso
from vpso.typing import Array1d, Array2d

from globopt.core.regression import Idw, Rbf, fit, partial_fit
from globopt.myopic.acquisition import acquisition
from globopt.util.random import make_seeds

logger = logging.getLogger(__name__)
CallbackT: TypeAlias = Callable[
    [int, Array1d, float, Array1d, float, float, Union[Idw, Rbf], Union[Idw, Rbf]], None
]


def _fit_mdl_to_init_points(
    mdl: Union[Idw, Rbf],
    func: Callable[[Array2d], ArrayLike],
    dim: int,
    lb: Array1d,
    ub: Array1d,
    X0: Union[int, Array2d],
    lhs_sampler: LatinHypercube,
) -> Union[Idw, Rbf]:
    """Sample, if necessary, the initial points, evaluate them and fits the regression
    model to them."""
    if isinstance(X0, int):
        X0 = (lb + (ub - lb) * lhs_sampler.random(X0))[np.newaxis]
    else:
        X0 = np.reshape(X0, (1, -1, dim))
    y0 = np.reshape(func(X0), (1, X0.shape[1], 1))
    return fit(mdl, X0, y0)


def _setup_vpso(
    lb: Array1d,
    ub: Array1d,
    seed: Optional[int],
    verbosity: int,
    pso_kwargs: Optional[dict[str, Any]],
) -> tuple[Array2d, Array2d, dict[str, Any], Optional[int]]:
    """Setup the bounds and kwargs for the VPSO algorithm."""
    lb = lb[np.newaxis]
    ub = ub[np.newaxis]
    if pso_kwargs is None:
        pso_kwargs = {}
    if "verbosity" not in pso_kwargs:
        pso_kwargs["verbosity"] = verbosity
    return lb, ub, pso_kwargs, pso_kwargs.pop("seed", seed)


def go(
    func: Callable[[Array2d], ArrayLike],
    lb: Array1d,
    ub: Array1d,
    #
    mdl: Union[Idw, Rbf],
    init_points: Union[int, Array2d] = 5,
    c1: float = 1.5078,
    c2: float = 1.4246,
    #
    maxiter: int = 50,
    #
    seed: Optional[int] = None,
    verbosity: int = logging.WARNING,
    callback: Optional[CallbackT] = None,
    #
    pso_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Array1d, float]:
    logger.setLevel(verbosity)

    # add initial points to regression model
    dim = lb.size
    lhs_sampler = LatinHypercube(d=dim, seed=seed)
    mdl = _fit_mdl_to_init_points(mdl, func, dim, ub, lb, init_points, lhs_sampler)

    # setup some quantities
    lb_, ub_, pso_kwargs, pso_seed = _setup_vpso(lb, ub, seed, verbosity, pso_kwargs)
    k = mdl.ym_.argmin()
    x_best = mdl.Xm_[0, k]
    y_best = mdl.ym_[0, k].item()
    if callback is not None:
        callback(0, x_best, y_best, np.nan, np.nan, np.nan, mdl, mdl)

    # main loop
    for i, seed_ in zip(range(1, maxiter + 1), make_seeds(pso_seed)):
        # choose next point to sample by minimizing the myopic acquisition function
        dym = mdl.ym_.ptp((1, 2), keepdims=True)
        x_new, acq_opt, _ = vpso(
            lambda x: acquisition(x, mdl, None, dym, c1, c2)[:, :, 0],
            lb_,
            ub_,
            **pso_kwargs,
            seed=seed_,
        )

        # evaluate next point and update the best point found so far
        y_new = np.reshape(func(x_new), (1, 1, 1))
        y_new_item = y_new.item()
        if y_new_item < y_best:
            x_best = x_new[0]
            y_best = y_new_item
        if logger.level <= logging.INFO:
            logger.info("best values at iteration %i: %e", i, y_best)

        # partially fit the regression model to the new point
        mdl_new = partial_fit(mdl, x_new[np.newaxis], y_new)

        # call callback at the end of the iteration and update model
        if callback is not None:
            callback(i, x_best, y_best, x_new, y_new_item, acq_opt, mdl, mdl_new)
        mdl = mdl_new

    return x_best, y_best
