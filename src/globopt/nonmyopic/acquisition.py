"""
Implementation of the non-myopic acquisition function for RBF/IDW-based Global
Optimization, based on the myopic function in [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""


from typing import Any, Literal, Optional

import numpy as np
from scipy.stats.qmc import MultivariateNormalQMC
from vpso import vpso
from vpso.typing import Array1d, Array2d, Array3d

from globopt.core.regression import RegressorType, partial_fit, predict, repeat
from globopt.myopic.acquisition import acquisition as myopic_acquisition


def acquisition(
    x: Array3d,
    mdl: RegressorType,
    horizon: int,
    discount: float,
    c1: float = 1.5078,
    c2: float = 1.4246,
    #
    type: Literal["rollout", "mpc"] = "rollout",
    # mc_iters: int = 1024,
    # quasi_monte_carlo: bool = True,
    # common_random_numbers: bool = True,
    # control_variate: bool = True,
    #
    ub: Optional[Array1d] = None,  # only when `type == "rollout"`
    lb: Optional[Array1d] = None,  # only when `type == "rollout"`
    pso_kwarg: Optional[dict[str, Any]] = None,
    #
    #
    # seed: Optional[int] = None,
    check: bool = True,  # TODO: set to False
) -> Array3d:
    if check:
        assert mdl.Xm_.shape[0] == 1, "regression model must be non-batched"
        if type == "rollout":
            assert (
                ub is not None and lb is not None
            ), "upper and lower bounds must be provided for rollout"
            assert x.shape[1] == 1, "x must have only one time step for rollout"
        else:
            assert (
                x.shape[1] == horizon
            ), "x must have the same number of time steps as the horizon"
    if pso_kwarg is None:
        pso_kwarg = {}

    # initialize quantities
    n_samples = x.shape[0]
    a = np.zeros(n_samples, dtype=np.float64)
    mdl = repeat(mdl, n_samples)
    y_min = mdl.ym_.min(1, keepdims=True)
    y_max = mdl.ym_.max(1, keepdims=True)
    lb_: Array2d = lb[np.newaxis].repeat(n_samples, 0)  # type: ignore[index]
    ub_: Array2d = ub[np.newaxis].repeat(n_samples, 0)  # type: ignore[index]

    # loop through the horizon
    for h in range(horizon):
        # compute the next point to query. If the strategy is "mpc", then the next point
        # is just the next point in the trajectory. If the strategy is "rollout", then,
        # for h=0, the next point is the input, and for h>0, the next point is the
        # minimizer of the myopic acquisition function, i.e., base policy.
        dym = y_max - y_min
        if type == "mpc":
            x_next = x[:, h, np.newaxis, :]
        elif h == 0:  # type == "rollout"
            x_next = x
        else:  # type == "rollout"
            # minimize the acquisition function
            x_next = vpso(
                lambda x: myopic_acquisition(x, mdl, c1, c2, None, dym)[:, :, 0],
                lb_,
                ub_,
                **pso_kwarg,
                # TODO: pass a seed here.. Can be one of the quasi-random numbers
            )[0][:, np.newaxis, :]

        # predict the sampling of the next point
        y_hat = predict(mdl, x_next)

        # add to reward
        a_h = myopic_acquisition(x_next, mdl, c1, c2, y_hat, y_max - y_min)
        a += (discount**h) * a_h[:, 0, 0]

        # fit regression to new point, and update min/max
        mdl = partial_fit(mdl, x_next, y_hat)
        y_min = np.minimum(y_min, y_hat)
        y_max = np.maximum(y_max, y_hat)
    return a
