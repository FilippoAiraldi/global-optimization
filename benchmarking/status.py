"""Status of an on-going benchmarking on synthetic problems."""


import argparse
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
from prettytable import PrettyTable

from globopt.core.problems import (
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
)

PROBLEMS = get_available_benchmark_problems() + get_available_simple_problems()


def get_status(filename_for_status: str) -> dict[str, dict[int, int]]:
    """Computes the status of the run.

    Parameters
    ----------
    filename : str
        filename of the results whose status needs to be retrieved.

    Returns
    -------
    dict[str, dict[int, int]]
        Returns the status of the run, i.e., the number of iterations for each problem
        and for each horizon that has already been computed.
    """
    with open(filename_for_status) as f:
        lines = f.readlines()  # better to read all at once

    out: dict[str, dict[int, int]] = {}
    maxiters = {n: get_benchmark_problem(n)[1] + 1 for n in PROBLEMS}
    for line in lines:
        elements = line.split(",")
        name = elements[0]
        horizon = int(elements[1])

        iters = len(elements[3:])
        assert iters == maxiters[name], (
            f"Incorrect number of iterations for {name}; expected {maxiters[name]}, got"
            f" {iters} instead."
        )

        if name not in out:
            out[name] = {}
        if horizon not in out[name]:
            out[name][horizon] = 0
        out[name][horizon] += 1
    return out


def filter_tasks_by_status(
    tasks: Iterator[tuple[int, str, int]], filename_for_status: str
) -> Iterable[tuple[int, str, int]]:
    """Filters the given tasks by the status of the given run.

    Parameters
    ----------
    tasks : Iterator of (int, str, int)
        Iterator of tasks to be filtered, i.e., (trial, problem name, horizon) tuples.
    filename : str
        The filename of the results whose status needs to be retrieved.

    Yields
    ------
    Iterable of (int, str, int)
        Filtered tasks that have not been completed yet.
    """
    status = get_status(filename_for_status)
    for trial, name, horizon in tasks:
        if (
            (name not in status)
            or (horizon not in status[name])
            or (trial >= status[name][horizon])  # trail is 0-based, status is a counter
        ):
            yield trial, name, horizon


def print_status(data: dict[str, dict[int, int]], tabletitle: Optional[str]) -> None:
    """Prints the status of the data in the given dictionary."""

    problem_names = set(data.keys())
    problem_names.update(PROBLEMS)
    problem_names = sorted(problem_names)

    horizons = set()
    for h in data.values():
        horizons.update(h.keys())
    horizons = sorted(horizons)

    table = PrettyTable()
    table.field_names = ["Function name"] + [f"h={h}" for h in horizons]
    if tabletitle is not None:
        table.title = tabletitle
    for problem_name in problem_names:
        name_data = data.get(problem_name)
        if name_data is None:
            row_data = ["-" for _ in horizons]
        else:
            row_data = [name_data.get(h, 0) for h in horizons]
        table.add_row([problem_name] + row_data)
    print(table)


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Displays the status of an on-going synthetic benchmark run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="Filenames of the results to be visualized.",
    )
    args = parser.parse_args()

    # load each result and plot
    for filename in args.filenames:
        print_status(get_status(filename), f"{filename} @ {datetime.now()}")
    plt.show()
