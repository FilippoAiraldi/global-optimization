"""Status of an on-going benchmarking on synthetic problems."""


import argparse
from typing import Optional

import matplotlib.pyplot as plt
from prettytable import PrettyTable

from globopt.core.problems import (
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
)

PROBLEMS = get_available_benchmark_problems() + get_available_simple_problems()


def load_data(filename: str) -> dict[str, dict[int, int]]:
    """Loads the count data from the given file."""
    with open(filename, "r") as f:
        lines = f.readlines()  # better to read all at once

    out: dict[str, dict[int, int]] = {}
    maxiters = {n: get_benchmark_problem(n)[1] for n in PROBLEMS}
    for line in lines:
        elements = line.split(",")
        name = elements[0]
        horizon = int(elements[1])

        iters = len(elements[3:])
        expected_iters = maxiters[name] + 1
        assert iters == expected_iters, (
            f"Incorrect number of iterations for {name}; expected {expected_iters}, got"
            f" {iters} instead."
        )

        if name not in out:
            out[name] = {}
        if horizon not in out[name]:
            out[name][horizon] = 0
        out[name][horizon] += 1
    return out


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
    include_title = len(args.filenames) > 1
    for filename in args.filenames:
        print_status(load_data(filename), filename if include_title else None)
    plt.show()
