"""
Visualization and summary of results of benchmarking of myopic and non-myopic Global
Optimization strategies on synthetic problems.
"""


import argparse
from itertools import zip_longest
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from prettytable import PrettyTable
from typing_extensions import TypeAlias
from vpso.typing import Array1d, Array2d

from globopt.core.problems import get_benchmark_problem

plt.style.use("bmh")

DataT: TypeAlias = dict[str, dict[int, tuple[Array2d, Array1d]]]


def load_data(filename: str) -> DataT:
    """Loads the data from the given file."""
    with open(filename, "r") as f:
        lines = f.readlines()  # better to read all at once

    out: dict[str, dict[int, tuple[list[list[str]], list[str]]]] = {}
    for line in lines:
        elements = line.split(",")
        name = elements[0]
        horizon = int(elements[1])
        cost = elements[2]
        bests = elements[3:]

        if name not in out:
            out[name] = {}
        if horizon not in out[name]:
            out[name][horizon] = ([], [])
        bin = out[name][horizon]
        bin[0].append(bests)
        bin[1].append(cost)

    for problem_data in out.values():
        for horizon, str_data in problem_data.items():
            problem_data[horizon] = tuple(np.asarray(o, dtype=float) for o in str_data)
    return out


def plot_results(data: DataT, figtitle: Optional[str]) -> None:
    """Plots the results in the given dictionary."""
    n_cols = 2
    n_rows = np.ceil(len(data) / n_cols).astype(int)
    fig, axs = plt.subplots(
        n_rows, n_cols, constrained_layout=True, figsize=(n_cols * 3.5, n_rows * 1.75)
    )
    axs = np.atleast_2d(axs)
    for i, (ax, problem_name) in enumerate(zip_longest(axs.flat, sorted(data.keys()))):
        if problem_name is None:
            # if there are more axes than results, set the axis off
            ax.set_axis_off()
            continue

        for horizon, (results, _) in data[problem_name].items():
            # compute the average, worst and best solution
            evals = np.arange(1, results.shape[1] + 1)
            avg = results.mean(axis=0)
            worst = results.max(axis=0)
            best = results.min(axis=0)

            # plot the results
            lbl = f"h={horizon}" if i == 0 else None
            c = ax.plot(evals, avg, label=lbl, lw=1.0)[0].get_color()
            # ax.plot(evals, worst, color=c, lw=0.25)
            # ax.plot(evals, best, color=c, lw=0.25)
            ax.fill_between(evals, worst, best, alpha=0.2, color=c)

        # plot the optimal point
        problem = get_benchmark_problem(problem_name)[0]
        fmin = np.full(results.shape[1], problem.f_opt)
        ax.plot(evals, fmin, "--", color="grey", zorder=-10000)
        ax.set_title(problem.f.__name__[1:], fontsize=11)
        ax.set_xlim(1, evals[-1])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    for ax in axs[-1]:
        ax.set_xlabel("number of function evaluations")
    fig.legend(loc="outside lower center", ncol=len(next(iter(data.values()))))
    fig.suptitle(figtitle, fontsize=12)
    # fig.savefig("results.pdf")


def plot_gaps(data: DataT, figtitle: Optional[str]) -> None:
    """Plots the gaps in the given dictionary."""
    n_cols = 2
    n_rows = np.ceil(len(data) / n_cols).astype(int)
    fig, axs = plt.subplots(
        n_rows, n_cols, constrained_layout=True, figsize=(n_cols * 3.5, n_rows * 1.75)
    )
    axs = np.atleast_2d(axs)
    for i, (ax, problem_name) in enumerate(zip_longest(axs.flat, sorted(data.keys()))):
        if problem_name is None:
            # if there are more axes than results, set the axis off
            ax.set_axis_off()
            continue

        problem = get_benchmark_problem(problem_name)[0]
        f_opt = problem.f_opt

        for horizon, (results, _) in data[problem_name].items():
            # compute the average, worst and best gaps
            evals = np.arange(1, results.shape[1] + 1)
            gaps = (results[:, 0, None] - results) / (results[:, 0, None] - f_opt)
            avg = gaps.mean(axis=0)
            worst = gaps.max(axis=0)
            best = gaps.min(axis=0)

            # plot the gaps
            lbl = f"h={horizon}" if i == 0 else None
            c = ax.plot(evals, avg, label=lbl, lw=1.0)[0].get_color()
            # ax.plot(evals, worst, color=c, lw=0.25)
            # ax.plot(evals, best, color=c, lw=0.25)
            ax.fill_between(evals, worst, best, alpha=0.2, color=c)

        # embellish
        ax.set_title(problem.f.__name__[1:], fontsize=11)
        ax.set_xlim(1, evals[-1])
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    for ax in axs[-1]:
        ax.set_xlabel("number of function evaluations")
    fig.legend(loc="outside lower center", ncol=len(next(iter(data.values()))))
    fig.suptitle(figtitle, fontsize=12)
    # fig.savefig("gaps.pdf")


def print_gaps_and_performance_summary(data: DataT, tabletitle: Optional[str]) -> None:
    """Prints the summary of the results in the given dictionary."""
    problem_names = sorted(data.keys())
    horizons = sorted(next(iter(data.values())).keys())
    f_opts = {n: get_benchmark_problem(n)[0].f_opt for n in problem_names}

    tables = (PrettyTable(), PrettyTable())
    field_names = ["Function name", ""] + [f"h={h}" for h in horizons]
    for table in tables:
        table.field_names = field_names
        if tabletitle is not None:
            table.title = tabletitle

    def populate_table(
        table: PrettyTable, get: Callable[[str, int], Array1d], order=np.argmax, prec=4
    ) -> None:
        for name in problem_names:
            means, medians = [], []
            for h in horizons:
                results = get(name, h)
                means.append(np.mean(results))
                medians.append(np.median(results))
            best_mean = order(means)
            best_median = order(medians)
            means_str = [f"{m:.{prec}f}" for m in means]
            medians_str = [f"{m:.{prec}f}" for m in medians]
            means_str[best_mean] = f"\033[1;34;40m{means_str[best_mean]}\033[0m"
            medians_str[best_median] = f"\033[1;34;40m{medians_str[best_median]}\033[0m"
            table.add_row([name, "mean"] + means_str)
            table.add_row(["", "median"] + medians_str)

    def get_gap(name: str, h: int) -> Array1d:
        results, _ = data[name][h]
        return (results[:, 0] - results[:, -1]) / (results[:, 0] - f_opts[name])

    populate_table(tables[0], get_gap)
    populate_table(tables[1], lambda name, h: data[name][h][1], order=np.argmin, prec=3)

    rows = zip(*(t.get_string().splitlines() for t in tables))
    print("\n".join(row1 + "\t" + row2 for row1, row2 in rows))


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Visualization of synthetic benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="Filenames of the results to be visualized.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print the summary and do not show the plots.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Only show the plot and do not print the summary.",
    )
    args = parser.parse_args()

    # load each result and plot
    include_title = len(args.filenames) > 1
    for filename in args.filenames:
        data = load_data(filename)
        title = filename if include_title else None
        if not args.no_plot:
            plot_results(data, title)
            plot_gaps(data, title)
        if not args.no_summary:
            print_gaps_and_performance_summary(data, title)
    plt.show()
