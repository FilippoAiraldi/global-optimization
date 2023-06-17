from functools import partial
from itertools import islice, product
from typing import Any, Iterable, Iterator

import numpy as np
from joblib import Parallel, delayed
from vpso.typing import Array1d, Array2d

from globopt.core.regression import RegressorType
from globopt.nonmyopic.acquisition import acquisition


def batched(iterable: Iterable[Any], n: int) -> Iterator[Any]:
    # taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def mpc_acquisition_by_brute_force(
    x: Array2d,
    mdl: RegressorType,
    h: int,
    c1: float,
    c2: float,
    discount: float,
    verbosity: int = 0,
    chunk_size: int = 2**11,
) -> Array1d:
    """Utility to compute the optimal acquisition function by brute force search."""
    # enumerate all trajectories of length h (n_trajectories, h, dim), evaluate the
    # acquisition function for each trajectory, and return the best for each x. The
    # computations are sped up by chunkifying the trajectories and evaluating the
    # acquisition function for each chunk in parallel

    def _batched(iterable, n: int):
        # taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    n_samples = x.shape[0]
    trajectories = product(*(x for _ in range(h)))
    chunks = map(
        np.asarray, _batched(trajectories, chunk_size)
    )  # n_traj = n_samples**h

    fun = partial(
        acquisition, mdl=mdl, horizon=h, discount=discount, c1=c1, c2=c2, type="mpc"
    )
    a_chunks = Parallel(n_jobs=-1, verbose=verbosity)(delayed(fun)(c) for c in chunks)
    a = np.concatenate(a_chunks, 0)
    return a.reshape(n_samples, n_samples ** (h - 1)).min(1)
