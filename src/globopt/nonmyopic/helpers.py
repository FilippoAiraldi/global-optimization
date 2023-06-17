from functools import partial
from itertools import islice, product
from typing import Any, Iterable, Iterator

import numpy as np
from joblib import Parallel, delayed
from vpso.typing import Array1d, Array3d

from globopt.core.regression import RegressorType
from globopt.nonmyopic.acquisition import deterministic_acquisition


def batched(iterable: Iterable[Any], n: int) -> Iterator[Any]:
    """Batch data into tuples of length n. The last batch may be shorter."""
    # taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def mpc_deterministic_acquisition_brute_force(
    x: Array3d,
    mdl: RegressorType,
    horizon: int,
    discount: float,
    c1: float,
    c2: float,
    verbosity: int = 0,
    chunk_size: int = 2**11,
) -> Array1d:
    """Utility to compute the optimal acquisition function by brute force search.
    Enumerates all trajectories of length `h` into an array of shape
    `(n_trajectories, h, dim)`, and then evaluates the acquisition function for each
    trajectory, and returns the best trajectory for each initial `x`. The computations
    are sped up by chunkifying the trajectories and evaluating the acquisition function
    for each chunk in parallel.

    Parameters
    ----------
    x : 3d array of shape (n_samples, 1, dim)
        The initial sampling points for which to compute the optimal deterministic
        acquisition function according to the MPC strategy. `n_samples` is the number
        of target points for which to compute the acquisition, and `dim` is the number
        of features/variables of each point
    mdl : Idw or Rbf
        Fitted model to use for computing the acquisition function.
    horizon : int
        Length of the prediced trajectory of sampled points.
    discount : float
        Discount factor for the lookahead horizon.
    c1 : float, optional
        Weight of the contribution of the variance function.
    c2 : float, optional
        Weight of the contribution of the distance function.
    verbosity : int, optional
        Verbosity of `joblib.Parallel`, by default `0`.
    chunk_size : int, optional
        Size of each processing chunk to be computed in parallel, by default 2**11.

    Returns
    -------
    1d array
        The deterministic non-myopic acquisition function for each target point, optimal
        with respect to the remaining of the trajectory.
    """
    x = x[:, 0, :]  # remove the horizon dimension
    n_samples = x.shape[0]
    trajectories = product(*(x for _ in range(horizon)))
    chunks = map(np.asarray, batched(trajectories, chunk_size))  # n_traj = n_samples**h
    fun = partial(
        deterministic_acquisition,
        mdl=mdl,
        horizon=horizon,
        discount=discount,
        c1=c1,
        c2=c2,
        type="mpc",
    )
    a_chunks = Parallel(n_jobs=-1, verbose=verbosity)(delayed(fun)(c) for c in chunks)
    a: np.ndarray = np.concatenate(a_chunks, 0)
    return a.reshape(n_samples, n_samples ** (horizon - 1)).min(1)
