from itertools import islice
from typing import Any, Iterable

import ray


def wait_tasks(refs: list[ray.ObjectRef], num_returns: int = 1) -> list[Any]:
    """Waits for a list of Ray object refs to complete and return the results."""
    # https://docs.ray.io/en/latest/ray-core/patterns/ray-get-submission-order.html
    results: list[Any] = []
    unfinished = refs
    while unfinished:
        finished, unfinished = ray.wait(
            unfinished, num_returns=min(num_returns, len(unfinished))
        )
        results.extend(ray.get(finished))
    return results


def batched(iterable: Iterable[Any], n: int) -> Iterable[tuple[Any, ...]]:
    """Batches data into tuples of length n. The last batch may be shorter.
    Taken from https://docs.python.org/3/library/itertools.html."""
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
