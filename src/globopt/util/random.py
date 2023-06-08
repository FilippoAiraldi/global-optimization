from hashlib import sha256
from itertools import count, repeat
from typing import Generator, Union

MAX_SEED = 2**32


def make_seed(s: str) -> int:
    """Generates a seed from a given string.

    Parameters
    ----------
    s : str
        The base string to generate the seed from.

    Returns
    -------
    int
        The generated seed.
    """
    seed = int.from_bytes(
        sha256(s.encode(), usedforsecurity=False).digest(), "big", signed=False
    )
    return seed % MAX_SEED


def make_seeds(
    seed: Union[None, str, int],
) -> Union[Generator[int, None, None], Generator[None, None, None]]:
    """Generates a stream of seeds from a given seed.

    Parameters
    ----------
    seed : int, str or None
        The base int or string to generate the seeds from.

    Yields
    ------
    Generator of ints or Nones
        A stream of seeds.
    """
    if seed is None:
        yield from repeat(None)
    else:
        generator = count(abs(seed) if isinstance(seed, int) else make_seed(seed))
        while True:
            yield next(generator) % MAX_SEED
