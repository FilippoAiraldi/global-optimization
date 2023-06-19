"""
Implementation of myopic Global Optimization strategy based on RBF or IDW regression.
The scheme was first proposed in [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats.qmc import LatinHypercube
from vpso import vpso
from vpso.typing import Array1d, Array2d

from globopt.core.regression import Idw, Rbf, fit, partial_fit
from globopt.myopic.acquisition import acquisition
from globopt.util.random import make_seeds


def _fit_mdl_to_init_points(
    mdl: Union[Idw, Rbf],
    func: Callable[[Array2d], ArrayLike],
    dim: int,
    lb: Array1d,
    ub: Array1d,
    X0: Union[int, Array2d],
    lhs_sampler: LatinHypercube,
) -> Union[Idw, Rbf]:
    """Samples, if necessary, the initial points, evaluate them and fits the regression
    model to them."""
    if isinstance(X0, int):
        X0 = (lb + (ub - lb) * lhs_sampler.random(X0))[np.newaxis]
    else:
        X0 = np.reshape(X0, (1, -1, dim))
    y0 = np.reshape(func(X0), (1, X0.shape[1], 1))
    return fit(mdl, X0, y0)


def _setup_vpso(
    lb: Array1d, ub: Array1d, seed: Optional[int], pso_kwargs: Optional[dict[str, Any]]
) -> tuple[Array2d, Array2d, dict[str, Any], Optional[int]]:
    """Sets up the bounds and kwargs for the VPSO algorithm."""
    lb = lb[np.newaxis]
    ub = ub[np.newaxis]
    if pso_kwargs is None:
        pso_kwargs = {}
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
    callback: Optional[Callable[[Literal["go", "nmgo"], dict[str, Any]], None]] = None,
    #
    pso_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Array1d, float]:
    """Global Optimization (GO) based on RBF or IDW regression [1].

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
    maxiter : int, optional
        Maximum number of iterations to run the algorithm for, by default `50`.
    seed : int, optional
        Seed for the random number generator, by default `None`.
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

    # add initial points to regression model
    dim = lb.size
    lhs_sampler = LatinHypercube(d=dim, seed=seed)
    mdl = _fit_mdl_to_init_points(mdl, func, dim, ub, lb, init_points, lhs_sampler)

    # setup some quantities
    lb_, ub_, pso_kwargs, pso_seed = _setup_vpso(lb, ub, seed, pso_kwargs)
    k = mdl.ym_.argmin()
    x_best = mdl.Xm_[0, k]
    y_best = mdl.ym_[0, k].item()
    if callback is not None:
        callback("go", locals())

    # main loop
    for _, seed_ in zip(range(maxiter), make_seeds(pso_seed)):
        # choose next point to sample by minimizing the myopic acquisition function
        dym = mdl.ym_.ptp(1, keepdims=True)
        x_new, acq_opt, _ = vpso(
            lambda x: acquisition(x, mdl, c1, c2, None, dym)[:, :, 0],
            lb_,
            ub_,
            **pso_kwargs,
            seed=seed_,
        )
        x_new = x_new[0]
        acq_opt = acq_opt.item()

        # evaluate next point and update the best point found so far
        y_new = np.reshape(func(x_new[np.newaxis]), ())
        if y_new < y_best:
            x_best = x_new[0]
            y_best = y_new.item()

        # partially fit the regression model to the new point
        mdl_new = partial_fit(mdl, x_new.reshape(1, 1, -1), y_new.reshape(1, 1, 1))

        # call callback at the end of the iteration and update model
        if callback is not None:
            callback("go", locals())
        mdl = mdl_new

    return x_best, y_best
