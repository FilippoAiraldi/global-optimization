from argparse import ArgumentTypeError
from datetime import datetime
from pathlib import Path
from warnings import warn

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
FCNTL_LOADED = True
try:
    import fcntl
except ImportError:
    FCNTL_LOADED = False
    warn("fcntl is unavailable; locking file for writing is impossible", ImportWarning)


def check_methods_arg(method: str) -> str:
    """Given a `method` argument, check its value."""
    if method in ["random", "ei", "myopic", "myopic-s"]:
        return method
    elif method.startswith("ms"):
        sampler_type, *fantasies = method[3:].split(".")
        if sampler_type != "gh" and sampler_type != "mc":
            raise ArgumentTypeError(
                f"Sampler type must be either `gh` or `mc`; got {sampler_type} instead."
            )
        if len(fantasies) == 0:
            raise ArgumentTypeError(
                "Multi-step methods must have at least one fantasy."
            )
        if not all(f.isdigit() and int(f) > 0 for f in fantasies):
            raise ArgumentTypeError(
                "Fantasies must be integers (greater than or equal to 0)."
            )
        return method
    else:
        raise ArgumentTypeError(f"Unrecognized method {method}.")


def fnv1a_64(s: str, base_seed: int = 0) -> int:
    """Creates a 64-bit hash of the given string using the FNV-1a algorithm."""
    hash64 = FNV_OFFSET + base_seed
    for char in s:
        hash64 ^= ord(char)
        hash64 *= FNV_PRIME
        hash64 &= 0xFFFFFFFFFFFFFFFF  # Ensure hash is 64-bit
    return hash64


def lock_write(filename: str, data: str) -> None:
    """Appends data to file while locking it to prevent concurrency (only for Linux)."""
    with open(filename, "a", encoding="utf-8") as f:
        if FCNTL_LOADED:
            try:
                fcntl.lockf(f, fcntl.LOCK_EX)
                f.write(data + "\n")
            finally:
                fcntl.lockf(f, fcntl.LOCK_UN)
        else:
            f.write(data + "\n")


def create_csv_if_needed(filename: str, header: str) -> str:
    """Creates the output csv file if it does not exist."""
    if filename is None or filename == "":
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    elif not filename.endswith(".csv"):
        filename += ".csv"
    if not Path(filename).is_file():
        lock_write(filename, header)
    return filename
