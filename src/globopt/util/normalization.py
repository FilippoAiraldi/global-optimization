"""Utility classes for easier normalization of ranges and values for problems."""


from typing import Union

import numpy as np
from vpso.typing import Array1d, Array2d

from globopt.core.problems import Problem


def forward(
    X: Array2d, lb: Array1d, ub: Array1d, lb_new: Array1d, ub_new: Array1d
) -> Array2d:
    """Normalizes `X` from the old bounds to the new bounds.

    Parameters
    ----------
    X : 2d array
        The array to normalize.
    lb, ub : 1d array
        The old lower and upper bounds.
    lb_new, ub_new : 1d array
        The new lower and upper bounds.

    Returns
    -------
    array
        The normalized array.
    """
    return (X - lb) / (ub - lb) * (ub_new - lb_new) + lb_new


def backward(
    X: Array2d, lb: Array1d, ub: Array1d, lb_new: Array1d, ub_new: Array1d
) -> Array2d:
    """Denormalizes `X` from the new bounds to the old bounds.

    Parameters
    ----------
    X : array
        The array to denormalize.
    lb, ub : array
        The old lower and upper bounds.
    lb_new, ub_new : array
        The new lower and upper bounds.

    Returns
    -------
    array
        The denormalized array.
    """
    return forward(X, lb_new, ub_new, lb, ub)


def normalize_problem(
    problem: Problem,
    lb_new: Union[float, Array1d] = -1.0,
    ub_new: Union[float, Array1d] = 1.0,
) -> Problem:
    """Created a normalized version of a problem.

    Parameters
    ----------
    problem : Problem
        The problem to normalize.
    lb_new, ub_new : float or array
        The new lower and upper bounds of the problem.

    Returns
    -------
    Problem
        The normalized problem.
    """
    f = problem.f
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub
    lb_new = np.broadcast_to(lb_new, dim)
    ub_new = np.broadcast_to(ub_new, dim)

    def normalized_f(x: Array2d) -> Array2d:
        return f(backward(x, lb, ub, lb_new, ub_new))

    return Problem(
        normalized_f,
        dim,
        lb_new,
        ub_new,
        forward(problem.x_opt, lb, ub, lb_new, ub_new),
    )
