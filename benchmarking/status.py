"""Status of an on-going benchmarking on problems."""


import argparse
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_status(csv: str) -> pd.DataFrame:
    """Computes the status of the run, given its csv file.

    Parameters
    ----------
    csv : str
        csv filename of the results whose status needs to be retrieved.

    Returns
    -------
    dict[str, dict[int, int]]
        Returns the status of the run, i.e., the number of iterations for each problem
        and for each horizon that has already been computed.
    """
    return (
        pd.read_csv(  # better to read all at once
            csv,
            sep=";",
            dtype={"problem": pd.StringDtype(), "method": pd.StringDtype()},
            usecols=["problem", "method"],
        )
        .groupby(["problem", "method"], dropna=False)
        .size()
    )


def filter_tasks_by_status(
    tasks: Iterator[tuple[int, str, str]], csv: str
) -> Iterable[tuple[int, str, int]]:
    """Yields tasks filtering out the already completed ones (taken from the given csv).

    Parameters
    ----------
    tasks : Iterator of (int, str, str)
        Iterator of tasks to be filtered, i.e., (trial, problem name, method) tuples.
    csv : str
        The filename of the csv whose status needs to be retrieved.

    Yields
    ------
    Iterable of (int, str, str)
        Filtered tasks that have not been completed yet.
    """
    # load the file into a dataframe, then loop over all the tasks. If the problem name
    # does not appear, or if it has never been computed for that method, or if its
    # number of trials is less than the one specified, then yield the task.
    if not Path(csv).is_file():
        yield from tasks
    else:
        df = get_status(csv)
        if df.empty:
            yield from tasks
        else:
            index = df.index
            for trial, name, method in tasks:
                # trail is 0-based, df holds 1-based counters instead
                if (name, method) not in index or df.loc[(name, method)] <= trial:
                    yield trial, name, method


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
        df = get_status(filename).to_string(na_rep="-")
        print(filename, "@", datetime.now(), "\n", df, "\n")
