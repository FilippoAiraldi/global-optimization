"""
Implementation of myopic Global Optimization strategy based on RBF or IDW regression.
The scheme was first proposed in [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Any, Callable, Literal, Optional, Union

import numba as nb
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats.qmc import LatinHypercube
from vpso import vpso
from vpso.typing import Array1d, Array2d

from globopt.core.regression import Idw, Rbf, fit, nb_Idw, nb_Rbf, partial_fit
from globopt.myopic.acquisition import acquisition


def _initialize(
    mdl: Union[Idw, Rbf],
    func: Callable[[Array2d], ArrayLike],
    lb: Array1d,
    ub: Array1d,
    X0: Union[int, Array2d],
    seed: Union[None, int, np.random.Generator],
    pso_kwargs: Optional[dict[str, Any]],
) -> tuple[
    Union[Idw, Rbf],
    Array2d,
    Array2d,
    int,
    Array1d,
    float,
    float,
    float,
    dict[str, Any],
    np.random.Generator,
]:
    """Initializes the Global Optimization algorithm."""
    # samples, if necessary, the initial points, evaluate them and fits the regression
    dim = lb.size
    np_random = np.random.default_rng(seed)
    lhs_sampler = LatinHypercube(d=dim, seed=np_random)
    if isinstance(X0, int):
        X0 = (lb + (ub - lb) * lhs_sampler.random(X0))[np.newaxis]
    else:
        X0 = np.reshape(X0, (1, -1, dim))
    y0 = np.reshape(func(X0[0]), (1, X0.shape[1], 1))
    mdl_fitted = fit(mdl, X0, y0)

    # pick the best point found so far, and minimum and maximum of the observations
    k = mdl_fitted.ym_.argmin()
    x_best = mdl_fitted.Xm_[0, k]
    y_best = mdl_fitted.ym_[0, k].item()
    y_min = y_best
    y_max = mdl_fitted.ym_.max().item()
    return (
        mdl_fitted,
        lb[np.newaxis],
        ub[np.newaxis],
        dim,
        x_best,
        y_best,
        y_min,
        y_max,
        pso_kwargs or {},
        np_random,
    )


def _next_query_point(
    mdl: Union[Idw, Rbf],
    lb: Array2d,
    ub: Array2d,
    c1: float,
    c2: float,
    y_min: float,
    y_max: float,
    np_random: np.random.Generator,
    pso_kwargs: Optional[dict[str, Any]],
) -> tuple[Array1d, float]:
    """Computes the next point to query by minimizing the acquisition function."""
    dym = np.full((1, 1, 1), y_max - y_min)
    vpso_func = lambda x: acquisition(x, mdl, c1, c2, None, dym)[:, :, 0]
    x_new, acq_opt, _ = vpso(vpso_func, lb, ub, seed=np_random, **pso_kwargs)
    return x_new[0], acq_opt.item()


@nb.njit(
    [
        nb.types.Tuple((mdl_type, nb.float64[:], nb.float64, nb.float64, nb.float64))(
            mdl_type,
            nb.float64[:],
            nb.float64,
            nb.float64[:],
            nb.float64,
            nb.float64,
            nb.float64,
        )
        for mdl_type in (nb_Rbf, nb_Idw)
    ],
    cache=True,
    nogil=True,
)
def _advance(
    mdl: Union[Idw, Rbf],
    x_new: Array1d,
    y_new: float,
    x_best: Array1d,
    y_best: float,
    y_min: float,
    y_max: float,
) -> tuple[Union[Idw, Rbf], Array1d, float, float, float]:
    """Advances the algorithm, given the new observation."""
    if y_new < y_best:
        x_best = x_new
        y_best = y_new
    mdl_fitted = partial_fit(
        mdl, x_new[np.newaxis, np.newaxis], np.full((1, 1, 1), y_new)
    )
    y_min = min(y_min, y_new)
    y_max = max(y_max, y_new)
    return mdl_fitted, x_best, y_best, y_min, y_max


def go(
    func: Callable[[Array2d], ArrayLike],
    lb: Array1d,
    ub: Array1d,
    mdl: Union[Idw, Rbf],
    init_points: Union[int, Array2d] = 5,
    c1: float = 1.5078,
    c2: float = 1.4246,
    maxiter: int = 50,
    seed: Union[None, int, np.random.Generator] = None,
    callback: Optional[Callable[[Literal["go", "nmgo"], dict[str, Any]], None]] = None,
    pso_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Array1d, float]:
    """Global Optimization (GO) based on RBF or IDW regression [1].

    Parameters
    ----------
    func : callable with 2d array adn returns array_like
        The (possibly, unknown) function to be minimized via GO. It must take as input
        a 2d array of shape `(n_points, dim)` and return a 1d array_like of shape
        `(n_points,)`. Given the points at which the function must be
        evaluated, the callable outputs the value of the function at those points.
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
    mdl, lb, ub, _, x_best, y_best, y_min, y_max, pso_kwargs, np_random = _initialize(
        mdl, func, lb, ub, init_points, seed, pso_kwargs
    )
    if callback is not None:
        callback("go", locals())

    for iteration in range(1, maxiter + 1):
        x_new, acq_opt = _next_query_point(
            mdl, lb, ub, c1, c2, y_min, y_max, np_random, pso_kwargs
        )
        y_new = float(func(x_new[np.newaxis]))
        mdl_new, x_best, y_best, y_min, y_max = _advance(
            mdl, x_new, y_new, x_best, y_best, y_min, y_max
        )
        if callback is not None:
            callback("go", locals())
        mdl = mdl_new

    return x_best, y_best
