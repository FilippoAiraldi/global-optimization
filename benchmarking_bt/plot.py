"""
Visualization and summary of results of benchmarking of myopic and non-myopic Global
Optimization strategies on synthetic problems.
"""

import argparse
from ast import literal_eval
from dataclasses import dataclass
from math import ceil
from typing import Callable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from prettytable import PrettyTable

from globopt.problems import get_benchmark_problem

plt.style.use("bmh")


@dataclass(repr=False)
class Results:
    """Simulation results for a given problem and horizon."""

    rewards: np.ndarray  # stage rewards based on the myopic acquisition function
    bests: np.ndarray  # best-so-far along iterations
    f_opt: float = None  # optimal value of the problem
    gaps: np.ndarray = None  # optimality gaps


def load_data(csv_filename: str) -> dict[str, dict[int, Results]]:
    """Loads the data from the given file."""
    with open(csv_filename, encoding="utf-8") as f:
        lines = f.readlines()  # better to read all at once

    out: dict[str, dict[int, tuple[list[list[float]], list[list[float]]]]] = {}
    for line in lines:
        elements = line.strip("\n").split(";")
        name = elements[0]
        horizon = int(elements[1])
        rewards = literal_eval(elements[2])
        bests = literal_eval(elements[3])
        assert len(rewards) + 1 == len(bests), "Incorrect data detected."

        if name not in out:
            out[name] = {}
        if horizon not in out[name]:
            out[name][horizon] = ([], [])
        dst = out[name][horizon]
        dst[0].append(rewards)
        dst[1].append(bests)

    for problem_data in out.values():
        for horizon, (rewards_lists, bests_lists) in problem_data.items():
            problem_data[horizon] = Results(
                np.asarray(rewards_lists), np.asarray(bests_lists)
            )
    return dict(sorted(out.items()))


def compute_optimality_gaps(data: dict[str, dict[int, Results]]) -> None:
    """Computes the optimality gap of the given results."""
    for problem_name, horizon_data in data.items():
        f_opt = get_benchmark_problem(problem_name)[0].optimal_value
        for results in horizon_data.values():
            init_best = results.bests[:, 0, np.newaxis]
            results.f_opt = f_opt
            results.gaps = (results.bests - init_best) / (f_opt - init_best)


def plot(data: dict[str, dict[int, Results]], figtitle: Optional[str]) -> None:
    """Plots the results in the given dictionary. In particular, it plots the
    convergence to the optimum, and the evolution of the optimality gap."""
    # create figure
    n_cols = 3
    n_rows = ceil(len(data) / n_cols)
    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 2.5))

    # create main grid of plots
    plotted_horizons = set()
    main_grid = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3)
    for i, (problem_name, horizon_data) in enumerate(data.items()):
        row, col = i // n_cols, i % n_cols

        # in this quadrant, create a subgrid for the convergence plot and the gap plot
        subgrid = main_grid[row, col].subgridspec(2, 1, hspace=0.1)
        ax_opt = fig.add_subplot(subgrid[0])
        ax_gap = fig.add_subplot(subgrid[1], sharex=ax_opt)

        for horizon, result in horizon_data.items():
            # plot the best convergence and the optimality gap evolution
            plotted_horizons.add(horizon)
            c = f"C{horizon - 1}"
            evals = np.arange(result.bests.shape[1])
            for ax, attr in zip((ax_opt, ax_gap), ("bests", "gaps")):
                data = getattr(result, attr)
                avg = data.mean(axis=0)
                worst = data.max(axis=0)
                best = data.min(axis=0)
                ax.plot(evals, avg, lw=1.0, color=c)
                # ax.plot(evals, worst, color=c, lw=0.25)
                # ax.plot(evals, best, color=c, lw=0.25)
                ax.fill_between(evals, worst, best, alpha=0.2, color=c)

        # plot also the optimal point in background
        fmin = np.full(evals.size, result.f_opt)
        ax_opt.plot(evals, fmin, "--", color="grey", zorder=-10000)

        # make axes pretty
        if col == 0:
            ax_opt.set_ylabel(r"$f^\star$")
            ax_gap.set_ylabel(r"$G$")
        if row == n_rows - 1:
            ax_gap.set_xlabel("Evaluations")
        ax_opt.set_title(problem_name, fontsize=11)
        ax_opt.tick_params(labelbottom=False)
        ax_opt.set_xlim(1, evals[-1])
        ax_gap.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    # crete legend manually
    handles = [
        Line2D([], [], label=f"h={h}", color=f"C{h-1}") for h in plotted_horizons
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(figtitle, fontsize=12)


def summarize(data: dict[str, dict[int, Results]], tabletitle: Optional[str]) -> None:
    """Prints the summary of the results in the given dictionary as two tables, one
    containing the (final) optimality gap, and the other the cumulative rewards."""
    # create tables
    problem_names = data.keys()
    horizons = sorted(
        {h for horizon_data in data.values() for h in horizon_data.keys()}
    )
    tables = (PrettyTable(), PrettyTable())
    field_names = ["Function name", ""] + [f"h={h}" for h in horizons]
    for table in tables:
        table.field_names = field_names
        if tabletitle is not None:
            table.title = tabletitle

    def populate_table(
        table: PrettyTable, getter: Callable[[Results], np.ndarray], precision: int = 4
    ) -> None:
        for name in problem_names:
            means, medians = [], []
            for h in horizons:
                quantity = getter(data[name][h])
                means.append(quantity.mean())
                medians.append(np.median(quantity))
            best_mean = np.argmax(means)
            best_median = np.argmax(medians)
            means_str = [f"{m:.{precision}f}" for m in means]
            medians_str = [f"{m:.{precision}f}" for m in medians]
            means_str[best_mean] = f"\033[1;34;40m{means_str[best_mean]}\033[0m"
            medians_str[best_median] = f"\033[1;34;40m{medians_str[best_median]}\033[0m"
            table.add_row([name, "mean"] + means_str)
            table.add_row(["", "median"] + medians_str)

    populate_table(tables[0], lambda res: res.gaps[:, -1])
    populate_table(tables[1], lambda res: res.rewards.sum(axis=1))

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
        title = filename if include_title else None
        loaded_data = load_data(filename)
        compute_optimality_gaps(loaded_data)
        if not args.no_plot:
            plot(loaded_data, title)
        if not args.no_summary:
            summarize(loaded_data, title)
    plt.show()
